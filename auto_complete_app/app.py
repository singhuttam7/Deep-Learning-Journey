import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LSTM Autocomplete AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD MODEL AND FILES
# ============================================================

@st.cache_resource
def load_resources():

    model = load_model(
        "lstm_autocomplete.h5",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


try:

    model, tokenizer, max_len = load_resources()

except Exception as e:

    st.error("Unable to load the trained model or required files.")
    st.code(str(e))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("✍️ LSTM Autocomplete AI")

st.subheader(
    "Intelligent word and sentence completion using LSTM"
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🧠 Model",
        value="LSTM"
    )

with col2:
    st.metric(
        label="📚 Vocabulary",
        value=len(tokenizer.word_index)
    )

with col3:
    st.metric(
        label="📏 Max Sequence",
        value=max_len
    )


st.divider()


# ============================================================
# TEXT AUTOCOMPLETE
# ============================================================

st.header("🚀 Text Autocomplete")

st.write(
    "Enter a word or sentence and let the LSTM model "
    "predict what should come next."
)


# ============================================================
# INPUT
# ============================================================

user_text = st.text_area(
    "Enter your text",
    placeholder="Example: machine learning is",
    height=120
)


# ============================================================
# MODE SELECTION
# ============================================================

mode = st.radio(
    "Choose prediction mode",
    [
        "🔤 Predict Single Word",
        "📝 Complete Sentence"
    ],
    horizontal=True
)


# ============================================================
# AUTOCOMPLETE FUNCTION
# ============================================================

def predict_next_word(text):

    # Convert text into token numbers
    sequence = tokenizer.texts_to_sequences([text])

    # No known words
    if len(sequence[0]) == 0:
        return None

    # Keep latest words
    sequence = sequence[0][-max_len + 1:]

    # Pad sequence
    sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    # Model prediction
    prediction = model.predict(
        sequence,
        verbose=0
    )

    # Highest probability index
    predicted_index = np.argmax(prediction[0])

    # Convert index back to word
    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            return word

    return None


# ============================================================
# SENTENCE GENERATION
# ============================================================

def generate_sentence(seed_text, number_of_words):

    result = seed_text.strip()

    for _ in range(number_of_words):

        next_word = predict_next_word(result)

        if next_word is None:
            break

        result += " " + next_word

    return result


# ============================================================
# SINGLE WORD MODE
# ============================================================

if mode == "🔤 Predict Single Word":

    st.subheader("Predict the Next Word")

    if st.button(
        "✨ Predict Next Word",
        use_container_width=True
    ):

        if user_text.strip() == "":

            st.warning(
                "Please enter some text first."
            )

        else:

            with st.spinner("AI is thinking..."):

                next_word = predict_next_word(
                    user_text
                )

            if next_word:

                st.success(
                    "Prediction completed!"
                )

                st.info(
                    f"Predicted Next Word:  **{next_word}**"
                )

            else:

                st.warning(
                    "Unable to predict the next word."
                )


# ============================================================
# SENTENCE MODE
# ============================================================

else:

    st.subheader("Complete Your Sentence")

    number_of_words = st.slider(
        "Number of words to generate",
        min_value=1,
        max_value=20,
        value=5
    )

    if st.button(
        "🚀 Complete Sentence",
        use_container_width=True
    ):

        if user_text.strip() == "":

            st.warning(
                "Please enter some text first."
            )

        else:

            with st.spinner(
                "Generating sentence..."
            ):

                generated_text = generate_sentence(
                    user_text,
                    number_of_words
                )

            st.success(
                "Sentence generated!"
            )

            st.info(
                f"Generated Text:\n\n{generated_text}"
            )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

with st.expander(
    "ℹ️ How does this application work?"
):

    st.subheader(
        "LSTM Autocomplete Pipeline"
    )

    st.write(
        "1. You enter a word or sentence."
    )

    st.write(
        "2. The tokenizer converts words into numerical tokens."
    )

    st.write(
        "3. The input is padded to the required sequence length."
    )

    st.write(
        "4. The trained LSTM predicts probabilities for the next word."
    )

    st.write(
        "5. The word with the highest probability is selected."
    )

    st.write(
        "6. For sentence completion, the predicted word is added to the input."
    )

    st.write(
        "7. The process repeats until the requested number of words is generated."
    )

    st.subheader("Files used")

    st.write(
        "`lstm_autocomplete.h5` → Trained LSTM model"
    )

    st.write(
        "`tokenizer.pkl` → Tokenizer"
    )

    st.write(
        "`max_len.pkl` → Maximum sequence length"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 LSTM Autocomplete AI | "
    "Built with Python • TensorFlow • Keras • Streamlit"
)

st.caption(
    "© 2026 LSTM Autocomplete Project"
)