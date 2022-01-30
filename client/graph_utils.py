import requests
import asyncio
from __future__ import annotations
from typing import List

class NodePipe:
  def __init__(self):
    pass

class Node:
  def __init__(self, url: str, method: str, outputs: List[Node] | List[NodePipe]):
    self.url = url
    self.method = method
    self.outputs = outputs

  async def sendAsyncRequest(self):
    retObj = None
    if self.method == "GET":
      retObj = requests.get(self.url).json()
    elif self.method == "POST":
      retObj = requests.post(self.url, data=self.payload).json()
    else:
      raise f"Fatal: {self.url} method {self.method} not found!"
    return retObj

class NodeFactory:
  def __init__(self, root_endpoint) -> None:
      self.root_endpoint = root_endpoint
      
  def createNode(self, method, path, input):
    return Node(self.root_endpoint + path)

class NodeOrchestrator:
  def __init__(self, filePath):
    pass
  