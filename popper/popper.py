from popper.utils import ExperimentalDataLoader, CustomDataLoader, DiscoveryBenchDataLoader
from popper.agent import SequentialFalsificationTest
from typing import Optional, Dict, Any
import os
import requests
import zipfile
import urllib
from tqdm import tqdm
import tarfile
import subprocess
import shutil

class Popper:
    """Wrapper class for hypothesis validation using sequential falsification testing."""
    
    def __init__(self, llm: str = "claude-3-5-sonnet-20240620", is_locally_served = False, server_port = None, api_key = "EMPTY", **kwargs):
        """Initialize Popper.
        
        Args:
            llm (str): Name of the LLM model to use
            **kwargs: Additional arguments to pass to SequentialFalsificationTest
        """
        self.llm = llm
        self.agent = None
        self.data_loader = None
        self.is_local = is_locally_served
        self.port = server_port
        self.api_key = api_key
        self.kwargs = kwargs

    def register_data(self, data_path: str, data_sampling: int = -1, loader_type: str = 'bio', metadata: Optional[Dict] = None):
        """Register data for hypothesis testing.
        
        Args:
            data_path (str): Path to data directory
            data_sampling (int): Number of datasets to sample (-1 for all)
            loader_type (str): Type of data loader to use ('bio', 'custom', or 'discovery_bench')
            metadata (Optional[Dict]): Metadata required for DiscoveryBenchDataLoader
        """            
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        self.data_path = data_path
        if loader_type == 'bio':
            if not os.path.exists(os.path.join(data_path, 'bio_database')):
                print('It will take a few minutes to download the data for the first time...')
                self.download_all_data()
            else:
                print('Bio data already exists, loading...')
            self.data_loader = ExperimentalDataLoader(
                data_path=data_path,
                table_dict_selection='all_bio',
                data_sampling=data_sampling
            )
        elif loader_type == 'bio_selected':
            if not os.path.exists(os.path.join(data_path, 'bio_database')):
                print('It will take a few minutes to download the data for the first time...')
                self.download_all_data()
            else:
                print('Bio data already exists, loading...')
            self.data_loader = ExperimentalDataLoader(
                data_path=data_path,
                table_dict_selection='default',
                data_sampling=data_sampling
            )
        elif loader_type == 'custom':
            print('Loading custom data...')
            self.data_loader = CustomDataLoader(data_folder=data_path)
        elif loader_type == 'discovery_bench':
            print('Loading discovery bench data...')
            if metadata is None:
                raise ValueError("Metadata must be provided for DiscoveryBenchDataLoader")
            self.data_loader = DiscoveryBenchDataLoader(data_path=data_path, metadata=metadata)
        else:
            raise ValueError(f"Unknown loader_type: {loader_type}")


    def configure(self, 
                 alpha: float = 0.1,
                 aggregate_test: str = 'E-value',
                 max_num_of_tests: int = 5,
                 max_retry: int = 5,
                 time_limit: int = 2,
                 relevance_checker: bool = True,
                 use_react_agent: bool = True):
        """Configure the sequential falsification test parameters.
        
        Args:
            alpha (float): Significance level
            aggregate_test (str): Test aggregation method
            max_num_of_tests (int): Maximum number of tests to run
            max_retry (int): Maximum number of retries for failed tests
            time_limit (int): Time limit in hours
            relevance_checker (bool): Whether to use relevance checker
            use_react_agent (bool): Whether to use ReAct agent
        """
        if self.data_loader is None:
            raise ValueError("Please register data first using register_data()")
            
        self.agent = SequentialFalsificationTest(llm=self.llm, is_local=self.is_local, port=self.port, api_key=self.api_key)
        self.agent.configure(
            data=self.data_loader,
            alpha=alpha,
            aggregate_test=aggregate_test,
            max_num_of_tests=max_num_of_tests,
            max_retry=max_retry,
            time_limit=time_limit,
            relevance_checker=relevance_checker,
            use_react_agent=use_react_agent,
            **self.kwargs
        )

    def validate(self, hypothesis: str) -> Dict[str, Any]:
        """Validate a scientific hypothesis using sequential falsification testing.
        
        Args:
            hypothesis (str): The scientific hypothesis to test
            
        Returns:
            Dict containing the test results including logs, final message, and parsed results
        """
        if self.agent is None:
            raise ValueError("Please configure the agent first using configure()")
            
        log, last_message, parsed_result = self.agent.go(hypothesis)
        
        return {
            "log": log,
            "last_message": last_message,
            "parsed_result": parsed_result
        }

    def _setup_default_agent(self):
        """Set up agent with default configuration if not already configured."""
        self.configure(
            alpha=0.1,
            aggregate_test='E-value',
            max_num_of_tests=5,
            max_retry=5,
            time_limit=2,
            relevance_checker=True,
            use_react_agent=True
        )
    
    def launch_UI(self):
        import gradio as gr
        from gradio import ChatMessage
        from time import time
        import asyncio
        import copy

        async def generate_response(prompt, 
                                    designer_history=[],
                                    executor_history=[],
                                    relevance_checker_history=[],
                                    error_control_history=[],
                                    summarizer_history=[]):

            #designer_history.append(ChatMessage(role="user", content=prompt))
            #yield designer_history, executor_history, relevance_checker_history, error_control_history, summarizer_history

            # Initialize log tracking
            prev_log = copy.deepcopy(self.agent.log)  # Store initial log state

            # Run the agent asynchronously
            task = asyncio.create_task(asyncio.to_thread(self.agent.go, prompt))

            while not task.done():  # Check while the agent is still running
                #print("Checking for new log messages...")
                await asyncio.sleep(1)  # Wait for 1 second

                # Check if log has changed
                if self.agent.log != prev_log:
                    prev_log = copy.deepcopy(self.agent.log)  # Update previous log state

                    # Convert new log messages to ChatMessage format
                    designer_msgs = [ChatMessage(role="assistant", content=msg) for msg in self.agent.log['designer']]
                    executor_msgs = [ChatMessage(role="assistant", content=msg) for msg in self.agent.log['executor']]
                    relevance_msgs = [ChatMessage(role="assistant", content=msg) for msg in self.agent.log['relevance_checker']]
                    sequential_msgs = [ChatMessage(role="assistant", content=msg) for msg in self.agent.log['sequential_testing']]
                    summarizer_msgs = [ChatMessage(role="assistant", content=msg) for msg in self.agent.log['summarizer']]

                    yield designer_msgs, executor_msgs, relevance_msgs, sequential_msgs, summarizer_msgs

            # Ensure final result is captured
            result = await task

            # Convert final logs to ChatMessage format before yielding
            designer_msgs = [ChatMessage(role="assistant", content=msg) for msg in self.agent.log['designer']]
            executor_msgs = [ChatMessage(role="assistant", content=msg) for msg in self.agent.log['executor']]
            relevance_msgs = [ChatMessage(role="assistant", content=msg) for msg in self.agent.log['relevance_checker']]
            sequential_msgs = [ChatMessage(role="assistant", content=msg) for msg in self.agent.log['sequential_testing']]
            summarizer_msgs = [ChatMessage(role="assistant", content=msg) for msg in self.agent.log['summarizer']]

            yield designer_msgs, executor_msgs, relevance_msgs, sequential_msgs, summarizer_msgs


        def like(evt: gr.LikeData):
            print("User liked the response")
            print(evt.index, evt.liked, evt.value)

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=1):
                    designer_chatbot = gr.Chatbot(label="Popper Experiment Designer", 
                                        type="messages", height=600, 
                                        show_copy_button=True, 
                                        show_share_button = True, 
                                        group_consecutive_messages = False,
                                        show_copy_all_button = True,
                    )
                    relevance_checker_chatbot = gr.Chatbot(label="Relevance Checker",
                                                type="messages", height=300,
                                                show_copy_button=True,
                                                show_share_button = True,
                                                group_consecutive_messages = False,
                                                show_copy_all_button = True
                    )

                with gr.Column(scale=1):
                    executor_chatbot = gr.Chatbot(label="Popper Experiment Executor", 
                                                type="messages", height=600, 
                                                show_copy_button=True, 
                                                show_share_button = True, 
                                                group_consecutive_messages = False, 
                                                show_copy_all_button = True,
                                                )
                    error_control_chatbot = gr.Chatbot(label="Sequential Error Control",
                                                type="messages", height=300,
                                                show_copy_button=True,
                                                show_share_button = True,
                                                group_consecutive_messages = False,
                                                show_copy_all_button = True
                    )

            with gr.Row():
                summarizer = gr.Chatbot(label="Popper Summarizer", 
                                                type="messages", height=300, 
                                                show_copy_button=True, 
                                                show_share_button = True, 
                                                group_consecutive_messages = False, 
                                                show_copy_all_button = True,
                                                )
            with gr.Row():
                # Textbox on the left, and Button with an icon on the right
                prompt_input = gr.Textbox(show_label = False, placeholder="What is your hypothesis?", scale=8)
                button = gr.Button("Validate")

            button.click(lambda: gr.update(value=""), inputs=None, outputs=prompt_input)
            # Bind button click to generate_response function, feeding results to both chatbots
            button.click(generate_response, inputs=[prompt_input, designer_chatbot, executor_chatbot, relevance_checker_chatbot, error_control_chatbot, summarizer], outputs=[designer_chatbot, executor_chatbot, relevance_checker_chatbot, error_control_chatbot, summarizer])

        demo.launch(share = True)


    def download_all_data(self):
        url = "https://dataverse.harvard.edu/api/access/datafile/10888484"
        file_name = 'popper_data_processed'
        self._download_and_extract_data(url, file_name)

    def _download_and_extract_data(self, url, file_name):
        """Download, extract, and merge directories using rsync."""
        tar_file_path = os.path.join(self.data_path, f"{file_name}.tar.gz")
        
        if not os.path.exists(tar_file_path):
            # Download the file
            print(f"Downloading {file_name}.tar.gz...")
            self._download_with_progress(url, tar_file_path)
            print("Download complete.")

            # Extract the tar.gz file
            print("Extracting files...")
            with tarfile.open(tar_file_path, 'r:gz') as tar:
                for member in tqdm(tar.getmembers(), desc="Extracting: "):
                    member.name = member.name.split('popper_data_processed/')[-1]  # Strip directory structure
                    tar.extract(member, self.data_path)           
                print("Extraction complete.")

    def _download_with_progress(self, url, file_path):
        """Download a file with a progress bar."""
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(request)
        total_size = int(response.getheader('Content-Length').strip())
        block_size = 1024  # 1 KB

        with open(file_path, 'wb') as file, tqdm(
            total=total_size, unit='B', unit_scale=True, desc="Downloading"
        ) as pbar:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                file.write(buffer)
                pbar.update(len(buffer))
