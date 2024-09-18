# LLM based Multi-Agent methods

Welcome to the LLM based Multi-Agent repository! This repository provides a lean implementation of cutting-edge techniques and methods for leveraging Large Language Models (LLMs) with multi-agent architectures for various tasks. This includes methods developed by Agnostiq Inc. as well as other state-of-the-art methods. The repository is designed to be modular and easy to use, allowing for quick experimentation and please use it with caution for production purposes.

## Quick example

> Note that we will be releasing a pip installable version soon. For now, please clone the repository and use it as a standalone package. 
>Step 1: `git clone https://github.com/AgnostiqHQ/multi-agent-llm.git`
>Step 2: `pip install -e .`

```python
import os; os.environ['OPENAI_API_KEY'] = "your_openai_api_key"

from multi_agent_llm import OpenAILLM, AIOT
from pydantic import BaseModel, Field

llm = OpenAILLM(model_name="gpt-4o-mini") # Currently we only have wrapper for OpenAI, but it can be easily extended to other LLMs

# Define the answer schema
class QueryAnswer(BaseModel):
    explanation: str = Field(description="Explanation of the answer")
    answer: str = Field(description="Final Answer")

# Initialize AIOT with the LLM and run a sample query
aiot = AIOT(llm=llm, iterations=2, answer_schema=QueryAnswer)
result = aiot.run("How many R's are there in the word 'Strawberry'?")

print(result)
```

```
Discussion Result
-----------------

How many R's are there in the word 'Strawberry'?

Thought Process:

Iteration 1:
IDA: Let's break down the word 'Strawberry'. First, count the letters in the word and then specifically look for the letter 'R'. Remember to pay attention to uppercase and lowercase letters, but focus primarily on the letter 'R'. How many instances of 'R' do you find?
LLMA:
{
  explanation: The word 'Strawberry' contains a total of three instances of the letter 'R'. This includes two uppercase 'R's and one lowercase 'r'. Therefore, when counting the letter 'R' in both forms, we find that there are 3 occurrences in total.
  answer: 3
}

Final Answer:
{
  explanation: The word 'Strawberry' contains a total of three instances of the letter 'R'. This includes two uppercase 'R's and one lowercase 'r'. Therefore, when counting the letter 'R' in both forms, we find that there are 3 occurrences in total.
  answer: 3
}
```

## Implemented Methods 

| **Method** | **Description**                                                                                                                                                                                                                                                       |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AIOT**   | Autonomous Iteration of Thought (AIOT) dynamically adapts its reasoning paths based on the evolving state of the conversation without generating alternate explorative thoughts that are ultimately discarded. [Quick Example](./iot/quick-example.ipynb) |
| **GIOT**   | Guided Iteration of Thought (GIOT) forces the LLM to continue iterating for a predefined number of steps, ensuring a thorough exploration of reasoning paths. [Quick Example](./iot/quick-example.ipynb)                                                  |



------

This repository also contains the results for the paper [Insert Paper Title Here]. You can find the relevant [experimental setups, datasets, and results](./examples/iot/exprements/). The folder contains results from various tasks. Feel free to explore these folders to reproduce the experiments or to get a deeper understanding of how the AIOT and GIOT frameworks work in practice.

