# 第18章：协作型 AI Agent 生态系统

随着 AI 技术的发展，单一的 AI Agent 已经无法满足复杂的实际需求。构建协作型 AI Agent 生态系统成为了必然趋势，这种生态系统中的多个 AI Agent 能够相互通信、协调和合作，共同完成复杂的任务。

## 18.1 Agent 间通信协议

设计一个简单但有效的 Agent 间通信协议是构建协作型 AI Agent 生态系统的基础。

```python
import json
from typing import Dict, Any

class Message:
    def __init__(self, sender: str, receiver: str, action: str, content: Dict[str, Any]):
        self.sender = sender
        self.receiver = receiver
        self.action = action
        self.content = content

    def to_json(self) -> str:
        return json.dumps({
            "sender": self.sender,
            "receiver": self.receiver,
            "action": self.action,
            "content": self.content
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        data = json.loads(json_str)
        return cls(data["sender"], data["receiver"], data["action"], data["content"])

class Agent:
    def __init__(self, name: str):
        self.name = name

    def send_message(self, receiver: str, action: str, content: Dict[str, Any]) -> Message:
        return Message(self.name, receiver, action, content)

    def receive_message(self, message: Message):
        print(f"{self.name} received message: {message.to_json()}")
        # 处理接收到的消息
        if message.action == "request_data":
            return self.send_message(message.sender, "data_response", {"data": "Some data"})
        elif message.action == "data_response":
            print(f"Received data: {message.content['data']}")
        # 可以添加更多的动作处理逻辑

# 使用示例
agent1 = Agent("Agent1")
agent2 = Agent("Agent2")

# Agent1 向 Agent2 发送请求
request_message = agent1.send_message("Agent2", "request_data", {"type": "user_info"})
agent2.receive_message(request_message)

# Agent2 回复 Agent1
response_message = agent2.send_message("Agent1", "data_response", {"data": "User info data"})
agent1.receive_message(response_message)
```

## 18.2 任务分配与协调机制

实现一个简单的任务分配和协调系统，使多个 Agent 能够协同工作。

```python
from typing import List, Dict
import random

class Task:
    def __init__(self, task_id: str, description: str, required_skills: List[str]):
        self.task_id = task_id
        self.description = description
        self.required_skills = required_skills
        self.assigned_agent = None
        self.status = "pending"

class SkillBasedAgent(Agent):
    def __init__(self, name: str, skills: List[str]):
        super().__init__(name)
        self.skills = skills
        self.current_task = None

    def can_perform_task(self, task: Task) -> bool:
        return all(skill in self.skills for skill in task.required_skills)

class TaskCoordinator:
    def __init__(self):
        self.tasks: List[Task] = []
        self.agents: List[SkillBasedAgent] = []

    def add_task(self, task: Task):
        self.tasks.append(task)

    def add_agent(self, agent: SkillBasedAgent):
        self.agents.append(agent)

    def assign_tasks(self):
        for task in self.tasks:
            if task.status == "pending":
                capable_agents = [agent for agent in self.agents if agent.can_perform_task(task) and agent.current_task is None]
                if capable_agents:
                    assigned_agent = random.choice(capable_agents)
                    task.assigned_agent = assigned_agent
                    task.status = "assigned"
                    assigned_agent.current_task = task
                    print(f"Task {task.task_id} assigned to {assigned_agent.name}")
                else:
                    print(f"No available agent for task {task.task_id}")

    def complete_task(self, task_id: str):
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task and task.status == "assigned":
            task.status = "completed"
            task.assigned_agent.current_task = None
            print(f"Task {task_id} completed by {task.assigned_agent.name}")

# 使用示例
coordinator = TaskCoordinator()

# 创建 Agents
agent1 = SkillBasedAgent("Agent1", ["python", "data_analysis"])
agent2 = SkillBasedAgent("Agent2", ["machine_learning", "nlp"])
agent3 = SkillBasedAgent("Agent3", ["python", "web_development"])

coordinator.add_agent(agent1)
coordinator.add_agent(agent2)
coordinator.add_agent(agent3)

# 创建任务
task1 = Task("T1", "Perform data analysis", ["python", "data_analysis"])
task2 = Task("T2", "Develop ML model", ["machine_learning"])
task3 = Task("T3", "Create web application", ["python", "web_development"])

coordinator.add_task(task1)
coordinator.add_task(task2)
coordinator.add_task(task3)

# 分配任务
coordinator.assign_tasks()

# 完成任务
coordinator.complete_task("T1")
coordinator.complete_task("T2")

# 再次分配任务
coordinator.assign_tasks()
```

## 18.3 集体智能与决策

实现一个基于集体智能的决策系统，综合多个 Agent 的意见来做出决策。

```python
from typing import List, Dict
import numpy as np

class DecisionMakingAgent(Agent):
    def __init__(self, name: str, expertise: float):
        super().__init__(name)
        self.expertise = expertise

    def make_decision(self, options: List[str]) -> Dict[str, float]:
        # 模拟决策过程，返回每个选项的评分
        scores = np.random.rand(len(options)) * self.expertise
        return dict(zip(options, scores))

class CollectiveIntelligenceSystem:
    def __init__(self, agents: List[DecisionMakingAgent]):
        self.agents = agents

    def make_collective_decision(self, options: List[str]) -> str:
        all_scores = []
        for agent in self.agents:
            scores = agent.make_decision(options)
            all_scores.append(scores)

        # 计算加权平均分数
        weighted_scores = {}
        for option in options:
            scores = [scores[option] * agent.expertise for agent, scores in zip(self.agents, all_scores)]
            weighted_scores[option] = sum(scores) / sum(agent.expertise for agent in self.agents)

        # 选择得分最高的选项
        best_option = max(weighted_scores, key=weighted_scores.get)
        return best_option

# 使用示例
agents = [
    DecisionMakingAgent("Expert1", expertise=0.9),
    DecisionMakingAgent("Expert2", expertise=0.7),
    DecisionMakingAgent("Expert3", expertise=0.8),
]

ci_system = CollectiveIntelligenceSystem(agents)

options = ["Option A", "Option B", "Option C"]
decision = ci_system.make_collective_decision(options)
print(f"The collective decision is: {decision}")
```

## 18.4 人机协作模式

设计一个人机协作系统，使 AI Agent 能够与人类用户进行有效的交互和协作。

```python
from typing import List, Dict
import random

class HumanAgent:
    def __init__(self, name: str):
        self.name = name

    def provide_input(self, prompt: str) -> str:
        return input(f"{self.name}, {prompt}: ")

class AIAssistant(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self.knowledge_base = {
            "python": "Python is a high-level programming language.",
            "machine_learning": "Machine learning is a subset of AI focused on learning from data.",
            "data_analysis": "Data analysis involves inspecting, cleaning, and modeling data."
        }

    def provide_information(self, topic: str) -> str:
        return self.knowledge_base.get(topic.lower(), "I don't have information on that topic.")

    def suggest_next_step(self, current_step: str) -> str:
        steps = {
            "problem_definition": "data_collection",
            "data_collection": "data_preprocessing",
            "data_preprocessing": "model_selection",
            "model_selection": "model_training",
            "model_training": "model_evaluation",
            "model_evaluation": "deployment"
        }
        return steps.get(current_step, "project_completion")

class HumanAICollaborationSystem:
    def __init__(self, human: HumanAgent, ai_assistant: AIAssistant):
        self.human = human
        self.ai_assistant = ai_assistant

    def collaborate_on_project(self, project_name: str):
        print(f"Starting collaboration on project: {project_name}")
        current_step = "problem_definition"

        while current_step != "project_completion":
            print(f"\nCurrent step: {current_step}")
            
            # AI提供信息
            ai_info = self.ai_assistant.provide_information(current_step)
            print(f"AI Assistant: Here's some information about {current_step}:\n{ai_info}")

            # 人类提供输入
            human_input = self.human.provide_input("What's your approach for this step?")
            print(f"Human input received: {human_input}")

            # AI建议下一步
            next_step = self.ai_assistant.suggest_next_step(current_step)
            print(f"AI Assistant: I suggest we move on to {next_step}.")

            # 人类确认
            confirmation = self.human.provide_input(f"Do you agree to move to {next_step}? (yes/no)")
            if confirmation.lower() == 'yes':
                current_step = next_step
            else:
                print("Human: Let's stay on the current step and refine our approach.")

        print(f"\nProject {project_name} completed successfully!")

# 使用示例
human = HumanAgent("Alice")
ai_assistant = AIAssistant("AI Assistant")
collaboration_system = HumanAICollaborationSystem(human, ai_assistant)

collaboration_system.collaborate_on_project("ML Model Development")
```

这个协作型 AI Agent 生态系统为更复杂、更智能的应用打下了基础。通过实现这些组件，我们可以创建能够相互协作、集体决策并与人类用户互动的 AI 系统。在实际应用中，你可能需要进一步完善和扩展这些概念，例如：

1. 实现更复杂的通信协议，支持加密和身份验证。
2. 开发更高级的任务分配算法，考虑工作负载平衡和优先级。
3. 改进集体决策机制，引入不同的投票或共识算法。
4. 增强人机协作系统，加入自然语言处理和情感分析能力。
5. 实现跨平台和分布式部署支持，使 Agent 能够在不同的环境中协作。

通过这些改进，我们可以构建更强大、更灵活的 AI Agent 生态系统，为各种复杂的实际应用提供解决方案。
