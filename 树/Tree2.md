
# Tree总结
### 遍历
关于树主要是遍历中的递归与迭代问题，以及顺序问题
```python
def traverse(self,root):
    #前序遍历位置  中左右 记录的位置是进入当前节点
    traverse(root.left)
    #中序遍历位置  左中右  是BST中很好的方法
    traverse(root.right)
    #后续遍历位置 左右中  记录的位置是出当前节点
```
是否传出```
traverse(root.left)``` 取决于递归函数是不是需要返回值给上一层，因为递归并不是直接跳出，而是将当且点（的值）返回上一层
eg：
```python
def Sumroot(self,root):
    # 明确当前节点的任务就是将自己的值和左子树的值和右子树的值相加
	if not root : return 0 # 终止条件
	val = root.val
    leftsum =  self.Sumroot(root.left) # 左子树
    rightsum = self.Sumroot(root.right) # 右子树
	return  leftsum+rightsum+val # 返回以当前节点为根节点的树的值之和
```
返回的是root的值之和
##### 还有个重要的点就是！ 
① 明确当前节点的任务
② 进入递归，相信他，不要跳入循环
③ 明确跳出循环条件
