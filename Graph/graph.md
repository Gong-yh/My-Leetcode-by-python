
# M 797 所有可能的路径（DAG有向无环图）
### 1. 记忆化递归，每一步都记录下来（已走路径）# 深度优先遍历
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
### 2. BFS # 广度优先遍历（层次遍历）
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


# M207 课程表
```python
from collections import defaultdict
from collections import deque
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # """
		# numCourses：必选课程数
		# prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
		# 	eg[[A,[B,C]]]：表示在选秀课程A之前一定要选秀课程[B,C]，而在选课程B之前又必须选C
			
		# 题目要求：判断是否能学完所有课程 eg：[[0，1],[1，0]] return:False
		# """
		# # 简单来说就是要判断是不是有环 bi --> ai 有向图 
		# # 判断有向无环图，需要用一个path记录一走路径 if A in path return False【错了，对于无向图才是这个关系】
        # # 还需要记录能修的课程数有没有达到numCourses
        # # [[1,0],[2,0],[3,2],[1,2]]
        # # 简化需求：因为没有要求返回路径，所以只要判断是不是存在一组数组中ai = bj 且 aj = bi【暴力求解，不太可行】
        # # 简化为从左往右 从右往左，是否存在一致list
        # d=collections.defaultdict(set)
        # for cur,pre in prerequisites:
        #     d[cur].add(pre)
        #     # print(10,d)
        #     d[cur]|=d[pre]
        #     # print(11,d)
            
        # for cur,pre in prerequisites[::-1]:
        #     d[cur].add(pre)
        #     d[cur]|=d[pre]
        #     # print(2,d)      
        # for cur,pre in prerequisites:
        #     if cur in d[pre]:
        #         return False
        # return True

        # """
        # 转换为拓扑图，判断有向无环图
        # 构建邻接矩阵
        # """
        # adjective = [[] for _ in range(numCourses)]
        # indegree = [0 for _ in range(numCourses)]
        # queue = deque()
        # for cur,pre in prerequisites:
        #     adjective[pre].append(cur) # 构建邻接矩阵
        #     indegree[cur]+=1 # 入度
        # for i in range(numCourses):
        #     if not indegree[i]:queue.append(i)
        # while queue:
        #     pre = queue.popleft()
        #     numCourses -= 1
        #     for cur in adjective[pre]:
        #         indegree[cur] -= 1
        #         if not indegree[cur] :queue.append(cur) # 没有入读了
        # return not numCourses

        """
        使用深度优先遍历DFS
        通过前序遍历记录走过的路线
        使用后序遍历回溯回去，当在遍历过程中，遇到在前序遍历中已经出现的节点，则判断为有环
        """
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        def dfs(i, adjacency, flags):
            if flags[i] == -1: return True
            if flags[i] == 1: return False
            flags[i] = 1
            for j in adjacency[i]:
                if not dfs(j, adjacency, flags): return False
            flags[i] = -1
            return True

        adjacency = [[] for _ in range(numCourses)]
        flags = [0 for _ in range(numCourses)]
        for cur, pre in prerequisites:
            adjacency[pre].append(cur)
        for i in range(numCourses):
            if not dfs(i, adjacency, flags): return False
        return True
```

# M 210 课程表Ⅱ
```python
from collections import deque
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # """
        # 用一个数组记录深度优先遍历顺序
        ## 这里理解错了，不能深度优先遍历，广度优先才是题目的思路
        # """
        # adjective = [[] for _ in range(numCourses)]
        # for cur,pre in prerequisites:
        #     adjective[pre].append(cur)
        # path = deque()
        # path.append(0)
        # pathlist = []
        # while path:
        #     cur = path.popleft()
        #     pathlist.append(cur)
        #     for nex in adjective[cur]:
        #         if nex in pathlist:return []
        #         path.append(nex)
        # return pathlist

        """
        用广度优先遍历，并记录走过的点
        """
        if numCourses == 0 :
            return []
        adjective = [[] for _ in range(numCourses)]
        indegree = [0 for _ in range(numCourses)]
        path = []
        pathlist = []
        for cur,pre in prerequisites:
            adjective[pre].append(cur)
            indegree[cur]+=1
        for cur in range(numCourses):
            if indegree[cur] == 0:
                path.append(cur)
        while path:
            top = path.pop(0)
            pathlist.append(top)

            for nex in adjective[top]:
                indegree[nex] -= 1
                if indegree[nex] == 0:
                    path.append(nex)
        if len(pathlist) != numCourses:
            return []
        return pathlist
```

# 785 判断二分图：
### 1. 广度优先遍历
```python
from collections import deque   #pycharm不在这导包直接用collections.deque会报错不知为何
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool: 
        # 广度优先遍历
        #color[i]标记i结点状态，0为未着色，1为着红色，-1为着蓝色
        color = [0 for _ in range(len(graph))]
        for i in range(len(graph)):
            #如果当前顶点未着色，以当前点为起始点bfs对当前连通图着色，起始点着红色
            if color[i] == 0:
                #对当前联通图顶点涂红色并入队
                color[i] = 1
                queue = deque()
                queue.append(i)
                #从当前顶点i开始bfs搜索当前连通图
                while queue:
                    cur = queue.popleft()
                    #注意是取cur的颜色而不是i的颜色！！！！！！写错了导致彻底错误还不好检查
                    col = color[cur]
                    #遍历当前顶点的所有邻居顶点       
                    for neighbor in graph[cur]:
                        #如果当前邻居顶点颜色与当前顶点相同，说明不是二分图，直接return False
                        if color[neighbor] == col:
                            return False
                        #如果当前邻居顶点颜色与当前顶点相反，说明当前邻居顶点已经入队过了，无需重复操作
                        elif color[neighbor] == -col:
                            continue
                        #如果当前邻居顶点未着色，着色为当前顶点相反颜色并入队以便后续搜索当前邻居顶点的邻居顶点
                        else:
                            color[neighbor] = -col
                            queue.append(neighbor)
        #所有顶点都着色完毕还没有发现相邻顶点颜色相同，说明是二分图
        return True

```
 ###  2. 深度优先遍历
 ```python
        vis = [0] * len(graph)
        def dfs(pos, color):
            vis[pos] = color
            for i in graph[pos]:
                # 颜色相同 or （未访问 且 继续深搜为False）
                # 可直接返回False
                if vis[i] == color or not (vis[i] or dfs(i, -color)):
                    return False
            return True

        # 不一定为联通图，需遍历每一个节点
        for i in range(len(graph)):
            if not vis[i] and not dfs(i, 1):
                return False
        return True
···
```
###### 参考 [labuladong的leetcode的图部分的一个讲解，还挺有用的](https://labuladong.gitee.io/algo/2/20/36/)

