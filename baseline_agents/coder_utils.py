# Set up the base template
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.tools import BaseTool
from langchain.chains.llm import LLMChain
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.schema import AgentAction, AgentFinish
from pydantic import Field
from typing import List, Union
import contextlib
import io
import logging
import re


logging.basicConfig(level=logging.INFO)


# set the maximum number of python code blocks that can be run
MAX_TURNS = 1

template = """{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: an input hypothesis that you must decide if it is True or False
Datasets: the names and descriptions of datasets relevant to the input hypothesis
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
WORKFLOW SUMMARY: this is the workflow that I used to find the final answer
Final Answer: True/False. Please output True if the input hypothesis is valid (e.g., you are able to reject the null hypothesis with statistical significance) and False if the input hypothesis is invalid (e.g., if you fail to reject the null hypothesis).

Please make sure the Final Answer is either True or False. Also generate a summary of the full workflow starting from data loading that led to the final answer as WORKFLOW SUMMARY:

IMPORTANT: all datasets have already been loaded into the global namespace as Pandas dataframes. You may access the data by referring to the EXACT dataframe names as provided in the "Datasets:" section.

NOTE: You will be able to execute the python code ONLY ONCE. So you will need to generate the complete code to solve the query in one go. Please provide the final answer after that. 

Begin!

{input}
{agent_scratchpad}"""


template_v2 = """{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: an input hypothesis that you must decide if it is True or False
Datasets: the names and descriptions of datasets relevant to the input hypothesis
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
WORKFLOW SUMMARY: this is the workflow that I used to find the final answer
Final Answer: True/False. Please output True if you believe the input hypothesis is correct and False if the input hypothesis is not based on your analysis.

Please make sure the Final Answer is either True or False. Also generate a summary of the full workflow starting from data loading that led to the final answer as WORKFLOW SUMMARY:

IMPORTANT: all datasets have already been loaded into the global namespace as Pandas dataframes. You may access the data by referring to the EXACT dataframe names as provided in the "Datasets:" section.

NOTE: You will be able to execute the python code ONLY ONCE. So you will need to generate the complete code to solve the query in one go. Please provide the final answer after that. 

Begin!

{input}
{agent_scratchpad}"""


def load_data_to_coder_globals(data_loader):
    for name, df in data_loader.table_dict.items():
            globals()[name] = df


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


# CustomOutputParser to parse the output of the LLM and execute actions
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            output = llm_output.split("Final Answer:")[-1].split()[0].strip().lower()
            if output not in ["true", "false", "yes", "no", "y", "n"]:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            return AgentFinish(
                return_values={"output": output in ["true", "yes", 'y']},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class CustomPythonAstREPLTool(PythonAstREPLTool):
    max_turns: int = Field(0, exclude=True)
    
    def _run(self, query: str, run_manager=None):
        if self.max_turns >= MAX_TURNS:
            return 'You cannot run the code more than once - you have already run it earlier. Please provide the "Final Answer:" immediately after "Thought:", based on whatever information you got till now. Do not attempt to output an "Action:" or run the code again.'
        self.max_turns += 1
        
        code_match = re.search(r"```(.*?)```", query, re.DOTALL)
        if code_match:
            # Extract code within backticks
            code = code_match.group(1)
        else:
            code = query
        code = code.strip()
        if code.startswith('"""'):
            code = code[3:].lstrip()
        if code.endswith('"""'):
            code = code[:-3].rstrip()
        if code.startswith("python"):
            code = code[len("python"):].lstrip()
        
        code = "import pandas as pd\n" + code
        
        exec_globals = globals().copy()
        exec_globals.update(__builtins__)
        
        output_capture = io.StringIO()
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            logging.getLogger().handlers[0].stream = output_capture
            try:
                exec(code, exec_globals)
            except Exception as e:
                return str(e)
        
        # Retrieve the output and return it
        output = output_capture.getvalue()
        return output if output else "Execution completed without output."

def create_agent(
    llm,
    handlers,
    max_iterations = None,
    early_stopping_method: str = "force",
    simple_template = False
):
    output_parser = CustomOutputParser()
    python_tool = CustomPythonAstREPLTool(callbacks=handlers)
    tools = [python_tool]
    tool_names = [tool.name for tool in tools]

    if simple_template:
        use_template = template_v2
    else:
        use_template = template

    print('Creating agent with template:', use_template)

    prompt = CustomPromptTemplate(
        template=use_template,
        tools=tools,
        input_variables=["system_prompt", "input", "intermediate_steps"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=handlers)

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        callbacks=handlers,
        early_stopping_method=early_stopping_method
    )