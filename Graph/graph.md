
### M 797 所有可能的路径（DAG有向无环图）
##### 记忆化递归，每一步都记录下来（已走路径）# 深度优先遍历
```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
	# 从起点开始，每个点都走一遍
		#深度优先遍历 一条路走到黑
		n = len(graph)
		def dfs(node):
			# 最后一个节点的邻居一定没有：(因为是无环)
			if node == n-1: #到达最后一个点
				return [[n-1]]
			ans = []
			for nex in graph[node]:# node --> nex --> res --> 最后一个（n-1）
				for res in dfs(nex)
				# 当前已走的路+接下来可以走的路的随便一个
					ans.append([node]+res)
				
			return ans
		return def(0)
```
##### BFS # 广度优先遍历（层次遍历）
```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
	# 广度优先遍历，层次遍历，栈的思想,先进先出
		n = len(graph)
		q = deque([[0]])
		ans = []
		while q:
			path = q.popleft()
			if path[-1] == n-1 :#已经走过最后一个了
				ans.append(path)
				continue
			for nex in graph(path[-1]): # 已经走过的最后一个 --> nex
				q.append(path+[nex])
			return ans

```
```python
print(q)
```
deque([[0, 1]])
deque([[0, 1], [0, 2]])
deque([[0, 2], [0, 1, 3]])
deque([[0, 1, 3], [0, 2, 3]])
