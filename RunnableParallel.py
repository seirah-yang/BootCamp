from langchain_core.runnables import Runnable
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class SimpleRunnable(Runnable):
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt

    def run(self, input_data):
        prompt_value = self.prompt.format(input_data)
        response = self.model(prompt_value)
        return response

model = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

simple_runnable = SimpleRunnable(model, prompt)
result = simple_runnable.run({"topic": "bears"})
print(result)
