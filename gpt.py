#!pip install langchain langchain-community langchain_openai

from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_openai import ChatOpenAI



examples = [
    {
        "question": "chest x - ray showing a large right - sided pleural effusion.",
        "answer": """
                        <chest X-ray>
                        Findings:
                        - Large pleural effusion in Rt. hemithorax.
                        - No active disease in both lungs.
                        - No cardiomegaly.
                        - No visible bony abnormality.

                        Conclusion:
                        1. Rt. pleural effusion, large in extent.
                        2. Otherwise, unremarkable.
                  """,
    },
    {
        "question": "chest x - ray showing a large left consolidation",
        "answer": """
                            <chest X-ray>
                            Findings:
                            - Large consolidation in Lt. lung.
                            - No cardiomegaly.
                            - No visible bony abnormality.

                            Conclusion:
                            1. Large consolidation in Lt. lung.
                            2. Otherwise, unremarkable.
                    """,
    },
    {
        "question": "chest x - ray showing cardiomegaly and pulmonary edema",
        "answer": """
                            <chest X-ray>
                            Findings:
                            - Pulmonary edema in both lungs.
                            - Cardiomegaly.
                            - No visible bony abnormality.

                            Conclusion:
                            1. Pulmonary edema in both lungs.
                            2. Cardiomegaly.
                    """,
    },
    {
        "question": "chest x - ray showing left rib fractures",
        "answer": """
                            <chest X-ray>
                            Findings:
                            - No active disease in both lungs.
                            - No cardiomegaly.
                            - Fractures in Lt. ribs.

                            Conclusion:
                            1. Lt. rib fractures.
                            2. Otherwise, unremarkable.
                    """,
    },
]

example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)
#print(example_prompt.format(**examples[0]))

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer:",
    input_variables=["question"],
)

def build_chain(key=" "):
        
        os.environ["OPENAI_API_KEY"] =key
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        chain = prompt | llm

        return chain

if __name__ == "__main__":

    your_key="sk"
    model=build_chain(key=your_key)

    print(model.invoke("chest x - ray showing cardiomegaly, pulmonary edema, and Left rib fractures").content)