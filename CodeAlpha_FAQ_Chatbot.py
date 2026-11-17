"""
FAQ Chatbot - CodeAlpha Internship Task 2
Simple chatbot that answers frequently asked questions using
NLTK for text processing and cosine similarity for matching.
"""

import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Keep the chatbot usable offline. When the NLTK corpora are already installed,
# use their full resources; otherwise use a small local stop-word list and let
# the lemmatizer fall back to the original token.
FALLBACK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "the", "to", "we",
    "what", "when", "where", "which", "who", "with", "you", "your",
}

# --- FAQ Knowledge Base ---
# i organized these as question-answer pairs
# covering a fictional online store scenario
FAQ_DATA = [
    {
        "question": "What are your shipping options?",
        "answer": "We offer Standard Shipping (5-7 business days), Express Shipping (2-3 business days), and Overnight Shipping. Free shipping on orders over $50!"
    },
    {
        "question": "How can I track my order?",
        "answer": "Once your order ships, you'll receive a tracking number via email. You can also check your order status in 'My Orders' section of your account."
    },
    {
        "question": "What is your return policy?",
        "answer": "You can return most items within 30 days of delivery for a full refund. Items must be unused and in original packaging. Some items like underwear and customized products are final sale."
    },
    {
        "question": "How do I create an account?",
        "answer": "Click 'Sign Up' on the top right corner, enter your email and create a password. You can also sign up using your Google or Facebook account."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, MasterCard, American Express, PayPal, Apple Pay, and Google Pay. All transactions are securely encrypted."
    },
    {
        "question": "How do I reset my password?",
        "answer": "Go to the login page and click 'Forgot Password'. Enter your email and we'll send you a reset link. The link expires in 24 hours."
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes! We ship to over 50 countries. International shipping usually takes 7-14 business days. Import duties and taxes may apply depending on your country."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can reach us via email at support@store.com, call us at 1-800-123-4567 (Mon-Fri 9AM-6PM EST), or use the live chat on our website."
    },
    {
        "question": "Can I cancel my order?",
        "answer": "You can cancel your order within 1 hour of placing it. After that, if it hasn't shipped yet, contact our support team and we'll try our best to cancel it."
    },
    {
        "question": "Do you have a loyalty program?",
        "answer": "Yes! Join our rewards program to earn points on every purchase. 1 point per dollar spent, and you can redeem 100 points for a $5 discount. Sign up is free!"
    },
    {
        "question": "Are your products authentic?",
        "answer": "Absolutely. We are an authorized retailer and all our products are 100% genuine. We source directly from brands and verified distributors."
    },
    {
        "question": "How do I apply a discount code?",
        "answer": "At checkout, you'll see a 'Promo Code' field. Enter your code there and click 'Apply'. The discount will be reflected in your order total."
    },
    {
        "question": "What if I receive a damaged item?",
        "answer": "We're sorry about that! Please contact support within 48 hours with photos of the damage. We'll arrange a replacement or full refund right away."
    },
    {
        "question": "Do you offer gift cards?",
        "answer": "Yes, we have digital gift cards available in $25, $50, $75, and $100 denominations. They never expire and can be used on any product."
    },
    {
        "question": "What are your store hours?",
        "answer": "Our online store is available 24/7. Our physical locations are open Monday-Saturday 10AM-9PM and Sunday 11AM-6PM."
    },
]


class FAQChatbot:
    def __init__(self, faq_data):
        self.faq_data = faq_data
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.stop_words = FALLBACK_STOP_WORDS

        # pre-process all the questions for matching
        self.questions = [item["question"] for item in faq_data]
        self.answers = [item["answer"] for item in faq_data]
        self.processed_questions = [self.preprocess(q) for q in self.questions]

        # build tfidf matrix from the faq questions
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)

    def preprocess(self, text):
        """clean up text - lowercase, remove punctuation, lemmatize"""
        text = text.lower()
        # get rid of punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # tokenize
        words = text.split()
        # remove stopwords and lemmatize
        cleaned = []
        for word in words:
            if word in self.stop_words:
                continue
            try:
                cleaned.append(self.lemmatizer.lemmatize(word))
            except LookupError:
                cleaned.append(word)
        words = cleaned
        return " ".join(words)

    def get_response(self, user_input):
        """find the best matching FAQ for the user's question"""
        processed_input = self.preprocess(user_input)

        if not processed_input.strip():
            return "Sorry, I didn't quite get that. Could you rephrase?"

        # transform user input using same vectorizer
        input_vec = self.vectorizer.transform([processed_input])

        # calculate similarity with all faq questions
        similarities = cosine_similarity(input_vec, self.tfidf_matrix).flatten()

        # get best match
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        # threshold - if similarity is too low, we don't have a good match
        if best_score < 0.15:
            return (
                "Hmm, I'm not sure about that. I can help with questions about "
                "shipping, returns, payments, orders, and account issues. "
                "Try asking something like 'What are your shipping options?'"
            )

        return self.answers[best_idx]

    def get_greeting(self):
        return (
            "Hi there! I'm the FAQ Bot for our online store.\n"
            "Ask me anything about shipping, returns, payments, "
            "orders, or your account.\n"
            "Type 'quit' or 'bye' to exit.\n"
        )


def show_sample_questions():
    """print some example questions the user can try"""
    print("\n--- Sample questions you can try ---")
    samples = [
        "How do I track my package?",
        "What's your refund policy?",
        "Do you ship internationally?",
        "How to reset password?",
        "What payment options are available?",
    ]
    for q in samples:
        print(f"  > {q}")
    print()


def main():
    bot = FAQChatbot(FAQ_DATA)

    print("=" * 50)
    print(bot.get_greeting())
    print("=" * 50)

    show_sample_questions()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # check for exit commands
        if user_input.lower() in ["quit", "exit", "bye", "q"]:
            print("Bot: Thanks for chatting! Have a great day!")
            break

        # check if user wants to see sample questions
        if user_input.lower() in ["help", "examples", "sample"]:
            show_sample_questions()
            continue

        response = bot.get_response(user_input)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
