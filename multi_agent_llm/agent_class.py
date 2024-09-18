from pydantic import BaseModel, Field


class Agent(BaseModel):
    name: str = Field(..., description="The name of the agent")
    role: str = Field(..., description="The role of the agent")
    function: str = Field(..., description="The function of the agent")
    
    def prompt(self, input_prompt: str, add_system_prompt: str = "") -> tuple:
        system_prompt = f"You are a: {self.name}. Your role: {self.role}. Your function: {self.function}. {add_system_prompt}"
        return system_prompt, f"{input_prompt}"