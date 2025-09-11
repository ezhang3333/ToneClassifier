import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification
from gpt_2_utils import prompt_zero, TOKENIZER, YES_ID, NO_ID, MODEL_NAME

# ----------------------------
# Load models once (cached)
# ----------------------------
@st.cache_resource
def load_seq2seq():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_clf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2ForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, pad_token_id=TOKENIZER.pad_token_id
    ).to(device)
    model.eval()
    return model, device

# ----------------------------
# Streamlit UI Config
# ----------------------------
st.set_page_config(page_title="GPT-2 Sentiment App", page_icon="🤖", layout="wide")

# Initialize session state for review history
if "history" not in st.session_state:
    st.session_state.history = []

# Main area
st.title("🤖 GPT-2 Sentiment Classifier")
st.write("Paste a review and choose how GPT-2 should classify it.")

# Input box (no validation highlight)
user_input = st.text_area("✍️ Enter your review:", "")

# Mode selector
mode = st.selectbox("⚙️ Choose model mode:", ["Zero-shot (Seq2Seq)", "Fine-tuned Classifier"])

# Prediction placeholder
prediction_area = st.empty()

if st.button("Classify"):
    if not user_input.strip():
        prediction_area.warning("Please enter some text.")
    else:
        if mode == "Zero-shot (Seq2Seq)":
            # Load zero-shot model
            lm, device = load_seq2seq()
            prompt = prompt_zero(user_input)
            inputs = TOKENIZER(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = lm(**inputs)
                logits = outputs.logits[0, -1]

            probs = torch.softmax(logits[[YES_ID, NO_ID]], dim=0).cpu().numpy()
            p_yes, p_no = probs
            sentiment = "Positive 😀" if p_yes > p_no else "Negative 😞"
            prediction_area.success(f"**Prediction (Zero-shot):** {sentiment}")
            st.write(f"Probabilities → Yes: {p_yes:.3f}, No: {p_no:.3f}")

        elif mode == "Fine-tuned Classifier":
            # Load classification head model
            lm_clf, device = load_clf()
            inputs = TOKENIZER(user_input, return_tensors="pt", truncation=True).to(device)

            with torch.no_grad():
                outputs = lm_clf(**inputs)
                logits = outputs.logits

            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            p_neg, p_pos = probs
            sentiment = "Positive 😀" if p_pos > p_neg else "Negative 😞"
            prediction_area.success(f"**Prediction (Classifier):** {sentiment}")
            st.write(f"Probabilities → Pos: {p_pos:.3f}, Neg: {p_neg:.3f}")

        # Save to history immediately
        st.session_state.history.append({
            "review": user_input,
            "prediction": sentiment
        })

# Sidebar = Review History (always pinned at top)
with st.sidebar:
    st.title("📜 Review History")
    pinned = st.container()  # this keeps it fixed at the top

    with pinned:
        if st.session_state.history:
            for i, item in enumerate(st.session_state.history[::-1], 1):
                st.markdown(f"**{i}.** _{item['review']}_ → {item['prediction']}")
        else:
            st.info("No reviews classified yet.")


# Footer
st.markdown("---")
st.markdown("💬 Powered by **GPT-2** · Zero-shot + Fine-tuned modes")