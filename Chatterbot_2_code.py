

import sys
import json
import random
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Computer Issue Chatbot')
        self.setGeometry(100, 100, 400, 400)

        self.layout = QVBoxLayout()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        self.user_input = QLineEdit()
        self.user_input.returnPressed.connect(self.send_message)

        self.send_button = QPushButton('Send')
        self.send_button.clicked.connect(self.send_message)

        self.layout.addWidget(self.chat_display)
        self.layout.addWidget(self.user_input)
        self.layout.addWidget(self.send_button)

        self.setLayout(self.layout)

        # Load knowledge base
        with open('knowledge_base.json', 'r') as file:
            knowledge_base = json.load(file)

        self.questions = []
        self.answers = []
        for item in knowledge_base['questions']:
            self.questions.append(item['question'])
            self.answers.append(item['answer'])

        # Preprocess data
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.questions)
        self.word_index = self.tokenizer.word_index
        self.sequences = self.tokenizer.texts_to_sequences(self.questions)
        self.padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(self.sequences, padding='post')

        # Define model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(self.word_index) + 1, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.questions), activation='softmax')
        ])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train model
        self.model.fit(self.padded_sequences, np.array(range(len(self.questions))), epochs=50)

    def get_response(self, user_input):
        sequence = self.tokenizer.texts_to_sequences([user_input])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post')
        prediction = self.model.predict(padded_sequence)[0]
        predicted_index = np.argmax(prediction)
        return self.answers[predicted_index]

    def send_message(self):
        user_input_text = self.user_input.text()
        if user_input_text.lower() == 'quit':
            self.chat_display.append("Goodbye!")
            sys.exit()
        response = self.get_response(user_input_text)
        self.chat_display.append("You: " + user_input_text)
        self.chat_display.append("Chatbot: " + response)
        self.user_input.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    chat_window = ChatWindow()
    chat_window.show()
    sys.exit(app.exec_())