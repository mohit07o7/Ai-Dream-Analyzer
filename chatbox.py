import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = [
    "I'm feeling great today!", "This is so frustrating.", "I'm really happy about the news.", "I'm feeling down and sad.",
    "Everything is going well.", "I'm so angry right now.", "I feel peaceful and calm.", "This is incredibly stressful.",
    "I'm excited for the weekend.", "I'm feeling anxious.", "The world is a beautiful place", "I am disappointed",
    "I love my friends", "I hate this", "I am very worried", "hello", "how are you?", "what is your name?",
    "tell me a joke", "that's funny", "i am bored", "what can I do?", "goodbye", "thank you", "no problem",
    "what is the weather?", "where are you from?", "what time is it?", "tell me about yourself"
]

labels = [
    "happy", "angry", "happy", "sad", "happy", "angry", "calm", "anxious", "excited", "anxious", "happy", "sad", "happy",
    "angry", "anxious", "greeting", "greeting", "identity", "joke", "response", "bored", "suggestion", "farewell",
    "gratitude", "acknowledgement", "weather", "identity", "time", "identity"
]

tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=30, padding="post", truncating="post")

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels = np.array(encoded_labels)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Embedding(2000, 32, input_length=30),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), verbose=1)

def predict_response(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=30, padding="post", truncating="post")
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label


def get_bot_response(predicted_label):
    responses = {
        "happy": ["That's wonderful!", "I'm glad to hear that!", "Keep smiling!"],
        "angry": ["Take a deep breath.", "Try to relax.", "I understand your frustration."],
        "sad": ["I'm here for you.", "It's okay to feel sad.", "Things will get better."],
        "calm": ["That's a good feeling.", "Enjoy the peace.", "Relax and unwind."],
        "anxious": ["Try some deep breathing exercises.", "It will be okay.", "Let's talk about it."],
        "excited": ["That's fantastic!", "I'm excited for you!", "Enjoy the moment!"],
        "greeting": ["Hello!", "Hi there!", "Greetings!"],
        "identity": ["I'm a helpful chatbot.", "I'm here to assist you.", "I'm a mood and general chat bot."],
        "joke": ["Why don't scientists trust atoms? Because they make up everything!", "What do you call a lazy kangaroo? Pouch potato."],
        "response": ["You're welcome!", "Glad I could help.", "Okay."],
        "bored": ["Try reading a book.", "Go for a walk.", "Listen to some music."],
        "suggestion": ["I suggest you relax", "Maybe try a new hobby","try meditating"],
        "farewell": ["Goodbye!", "See you later!", "Have a nice day!"],
        "gratitude": ["You're welcome!", "No problem!", "My pleasure."],
        "acknowledgement": ["Okay.", "Understood.", "Got it."],
        "weather": ["I cannot provide real time weather information","Check your local weather app"],
        "time": ["I cannot provide real time time information","Check your device clock"]
    }
    return np.random.choice(responses.get(predicted_label, ["I'm not sure how to respond to that."]))

print("Chatbot: Hello! How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    predicted_label = predict_response(user_input)
    bot_response = get_bot_response(predicted_label)
    print("Chatbot:", bot_response)