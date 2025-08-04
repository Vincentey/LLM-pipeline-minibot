from dotenv import load_dotenv
import os

load_dotenv()
openAikey = os.getenv("OPENAI_API_KEY")


from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableSequence

#summarization runnable
llm = ChatOpenAI(model= "gpt-4o")
summarize_prompt= ChatPromptTemplate.from_template("summarize this text: {text}")
summarizer = summarize_prompt|llm

#translation runnable
translation_prompt = ChatPromptTemplate.from_template("translates this text to German:{text}")
translator = translation_prompt|llm

#sentiment analyser
sentiment_prompt = ChatPromptTemplate.from_template("what is the sentiment of this text: {text}")
sentiment_analyser = sentiment_prompt|llm

chain = RunnableParallel(summary= summarizer,
                         translation= translator,
                         sentiment= sentiment_analyser)

passage= "Well done. But even when I'm working I prioritize getting us fed. This arrangement of starving before eating is going to predispose us to stomach ulcers. It cannot be said that there is an HK in Nigeria where people have not eaten since morning."
answer= chain.invoke({"text": passage})

print("summary:", answer["summary"].content)
print("translation:", answer["translation"].content)
print("sentiment:", answer["sentiment"].content)