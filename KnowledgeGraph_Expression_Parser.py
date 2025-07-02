#!/usr/bin/env python3
import json
import asyncio
import argparse
from typing import Dict, Any, List, Optional, Union
from enum import Enum

class TokenType(Enum):
    """Token 类型枚举"""
    LITERAL = "LITERAL"      # 数字
    IDENTIFIER = "IDENTIFIER" # 变量名
    OPERATOR = "OPERATOR"     # 运算符
    LPAREN = "LPAREN"         # 左括号
    RPAREN = "RPAREN"         # 右括号
    EOF = "EOF"               # 结束符

class Token:
    """词法单元"""
    def __init__(self, type_: TokenType, value: str, position: int = 0):
        self.type = type_
        self.value = value
        self.position = position
    
    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class Lexer:
    """词法分析器"""
    def __init__(self, text: str):
        self.text = text.replace(' ', '')  # 移除空格
        self.position = 0
        self.current_char = self.text[0] if text else None
    
    def advance(self):
        """移动到下一个字符"""
        self.position += 1
        if self.position >= len(self.text):
            self.current_char = None
        else:
            self.current_char = self.text[self.position]
    
    def read_number(self) -> str:
        """读取数字"""
        result = ''
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            result += self.current_char
            self.advance()
        return result
    
    def read_identifier(self) -> str:
        """读取标识符"""
        result = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        return result
    
    def tokenize(self) -> List[Token]:
        """词法分析"""
        tokens = []
        
        while self.current_char is not None:
            if self.current_char.isdigit():
                number = self.read_number()
                tokens.append(Token(TokenType.LITERAL, number, self.position))
            
            elif self.current_char.isalpha():
                identifier = self.read_identifier()
                tokens.append(Token(TokenType.IDENTIFIER, identifier, self.position))
            
            elif self.current_char == '(':
                tokens.append(Token(TokenType.LPAREN, '(', self.position))
                self.advance()
            
            elif self.current_char == ')':
                tokens.append(Token(TokenType.RPAREN, ')', self.position))
                self.advance()
            
            elif self.current_char in '+-*/':
                tokens.append(Token(TokenType.OPERATOR, self.current_char, self.position))
                self.advance()
            
            elif self.current_char == '^':
                tokens.append(Token(TokenType.OPERATOR, '^', self.position))
                self.advance()
            
            else:
                self.advance()
        
        tokens.append(Token(TokenType.EOF, '', self.position))
        return tokens

class ASTNode:
    """AST 节点基类"""
    pass

class NumberNode(ASTNode):
    """数字节点"""
    def __init__(self, value: str):
        self.value = value
    
    def __repr__(self):
        return f"NumberNode({self.value})"

class VariableNode(ASTNode):
    """变量节点"""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f"VariableNode({self.name})"

class MultiOpNode(ASTNode):
    """多元运算节点 - 支持同级运算符的多叉树结构"""
    def __init__(self, operands: List[ASTNode], operators: List[str], priority: int):
        self.operands = operands  # 操作数列表
        self.operators = operators  # 操作符列表
        self.priority = priority  # 优先级
    
    def __repr__(self):
        return f"MultiOpNode(operands={len(self.operands)}, ops={self.operators}, priority={self.priority})"

class ParenthesesNode(ASTNode):
    """括号节点 - 将括号作为独立的运算符处理"""
    def __init__(self, inner_expr: ASTNode):
        self.inner_expr = inner_expr  # 括号内的表达式
    
    def __repr__(self):
        return f"ParenthesesNode({self.inner_expr})"

class Parser:
    """递归下降解析器 - 改造为多叉树结构"""
    
    PRECEDENCE = {
        '+': 20, '-': 20,
        '*': 40, '/': 40,
        '^': 60
    }
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = tokens[0] if tokens else Token(TokenType.EOF, '')
    
    def advance(self):
        """移动到下一个 Token"""
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = Token(TokenType.EOF, '')
    
    def get_precedence(self, operator: str) -> int:
        """获取运算符优先级"""
        return self.PRECEDENCE.get(operator, -1)
    
    def parse_expression(self) -> ASTNode:
        """解析表达式（入口点）"""
        return self.parse_precedence_level(0)  # 从最低优先级开始
    
    def parse_precedence_level(self, min_precedence: int) -> ASTNode:
        """按优先级层次解析，构建多叉树"""
        left = self.parse_primary()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.get_precedence(self.current_token.value) >= min_precedence):
            
            # 收集相同优先级的操作
            operands = [left]
            operators = []
            current_precedence = self.get_precedence(self.current_token.value)
            
            # 收集所有相同优先级的操作
            while (self.current_token.type == TokenType.OPERATOR and 
                   self.get_precedence(self.current_token.value) == current_precedence):
                
                op = self.current_token.value
                operators.append(op)
                self.advance()
                
                # 解析右操作数，处理更高优先级
                right = self.parse_precedence_level(current_precedence + 1)
                operands.append(right)
            
            # 如果只有一个操作数，直接返回
            if len(operands) == 1:
                left = operands[0]
            else:
                # 创建多叉树节点
                left = MultiOpNode(operands, operators, current_precedence)
        
        return left
    
    def parse_primary(self) -> ASTNode:
        """解析基本表达式"""
        if self.current_token.type == TokenType.LITERAL:
            value = self.current_token.value
            self.advance()
            return NumberNode(value)
        
        elif self.current_token.type == TokenType.IDENTIFIER:
            name = self.current_token.value
            self.advance()
            return VariableNode(name)
        
        elif self.current_token.type == TokenType.LPAREN:
            self.advance()  # 跳过 '('
            expr = self.parse_expression()  # 递归解析括号内容
            if self.current_token.type != TokenType.RPAREN:
                raise SyntaxError(f"Expected ')' but got {self.current_token}")
            self.advance()  # 跳过 ')'
            # 将括号作为独立的运算符节点
            return ParenthesesNode(expr)
        
        elif self.current_token.type == TokenType.OPERATOR and self.current_token.value == '-':
            # 负号处理
            self.advance()
            operand = self.parse_primary()
            return MultiOpNode([NumberNode('0'), operand], ['-'], 20)
        
        else:
            raise SyntaxError(f"Unexpected token: {self.current_token}")

class KnowledgeGraphParser:
    """知识图谱解析器 - 多叉树结构支持同级节点批量处理"""
    
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.priority_levels = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    
    def parse_to_kg(self, expression: str) -> Dict[str, Any]:
        """将表达式解析为多叉树知识图谱格式"""
        # 词法分析
        lexer = Lexer(expression)
        tokens = lexer.tokenize()
        
        # 语法分析
        parser = Parser(tokens)
        ast_root = parser.parse_expression()
        
        # 保存 AST 结构
        import os
        os.makedirs('.cache', exist_ok=True)
        self._save_ast_structure(ast_root, ".cache/output.ast", expression)
        
        # 转换为知识图谱
        kg_root = self._ast_to_kg(ast_root, [0], 0, expression)
        
        return {"root": kg_root}
    
    def _save_ast_structure(self, ast_root: ASTNode, filename: str, original_expr: str):
        """保存 AST 结构到文件"""
        content = []
        content.append("# AST 解析结果 (多叉树)\n")
        content.append(f"## 原始表达式: {original_expr}\n\n")
        
        content.append("## AST 树形结构:\n```\n")
        self._format_ast_tree(ast_root, content, "")
        content.append("\n```\n\n")
        
        if self.debug_mode:
            content.append("## 调试信息:\n")
            content.append("### 解析器类型: 递归下降解析器（多叉树结构）\n")
            content.append("### 运算符优先级表:\n")
            content.append("```\n")
            for op, prec in Parser.PRECEDENCE.items():
                content.append(f"{op}: {prec}\n")
            content.append("```\n")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(''.join(content))
        
        if self.debug_mode:
            print(f"AST 结构已保存到: {filename}")
    
    def _format_ast_tree(self, node: ASTNode, lines: List[str], prefix: str):
        """格式化 AST 树形结构"""
        if isinstance(node, NumberNode):
            lines.append(f"NumberNode({node.value})\n")
        elif isinstance(node, VariableNode):
            lines.append(f"VariableNode({node.name})\n")
        elif isinstance(node, ParenthesesNode):
            lines.append(f"ParenthesesNode()\n")
            lines.append(f"{prefix}└── inner: ")
            self._format_ast_tree(node.inner_expr, lines, prefix + "    ")
        elif isinstance(node, MultiOpNode):
            lines.append(f"MultiOpNode(priority={node.priority}, ops={node.operators})\n")
            for i, operand in enumerate(node.operands):
                is_last = i == len(node.operands) - 1
                branch = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "
                lines.append(f"{prefix}{branch}operand[{i}]: ")
                self._format_ast_tree(operand, lines, prefix + next_prefix)
    
    def _ast_to_kg(self, node: ASTNode, kg_id: List[int], depth: int, original_expr: str) -> Dict[str, Any]:
        """将 AST 节点转换为知识图谱节点"""
        if isinstance(node, NumberNode):
            return {
                "KG_ID": kg_id,
                "expression": node.value,
                "is_variable": True,
                "operator": "",
                "depth": depth,
                "priority": 0
            }
        
        elif isinstance(node, VariableNode):
            return {
                "KG_ID": kg_id,
                "expression": node.name,
                "is_variable": True,
                "operator": "",
                "depth": depth,
                "priority": 0
            }
        
        elif isinstance(node, ParenthesesNode):
            # 处理括号节点
            child_id = kg_id + [0]
            child = self._ast_to_kg(node.inner_expr, child_id, depth + 1, original_expr)
            child["operator"] = "()"
            
            return {
                "KG_ID": kg_id,
                "expression": f"({self._extract_node_expression(node.inner_expr)})",
                "is_variable": False,
                "operator": "()",
                "depth": depth,
                "priority": 100,  # 括号具有最高优先级
                "children": [child]
            }
        
        elif isinstance(node, MultiOpNode):
            # 构建表达式字符串
            if depth == 0:
                expression = original_expr
            else:
                expression = self._build_expression(node)
            
            # 构建子节点（多叉树）
            children = []
            for i, operand in enumerate(node.operands):
                child_id = kg_id + [i]
                child_op = "" if i == 0 else node.operators[i-1]
                child = self._ast_to_kg(operand, child_id, depth + 1, original_expr)
                child["operator"] = child_op
                children.append(child)
            
            return {
                "KG_ID": kg_id,
                "expression": expression,
                "is_variable": False,
                "operator": "",
                "depth": depth,
                "priority": self.priority_levels.get(node.operators[0] if node.operators else '', 0),
                "children": children
            }
    
    def _build_expression(self, node: MultiOpNode) -> str:
        """从多叉树节点构建表达式字符串"""
        if len(node.operands) == 1:
            return self._extract_node_expression(node.operands[0])
        
        parts = [self._extract_node_expression(node.operands[0])]
        for i, op in enumerate(node.operators):
            if i + 1 < len(node.operands):
                parts.append(op + self._extract_node_expression(node.operands[i + 1]))
        
        return ''.join(parts)
    
    def _extract_node_expression(self, node: ASTNode) -> str:
        """提取节点的表达式字符串"""
        if isinstance(node, NumberNode):
            return node.value
        elif isinstance(node, VariableNode):
            return node.name
        elif isinstance(node, ParenthesesNode):
            return f"({self._extract_node_expression(node.inner_expr)})"
        elif isinstance(node, MultiOpNode):
            return self._build_expression(node)
        return "unknown"
    
    # 保持与 V3.0 兼容的方法
    def get_leaf_nodes(self, kg_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取所有叶子节点"""
        leaves = []
        self._collect_leaves(kg_data["root"], leaves)
        return leaves
    
    def _collect_leaves(self, node: Dict[str, Any], leaves: List):
        """递归收集叶子节点"""
        if "children" not in node:
            leaves.append(node)
        else:
            for child in node["children"]:
                self._collect_leaves(child, leaves)
    
    def get_nodes_by_depth(self, kg_data: Dict[str, Any]) -> Dict[int, List[Dict]]:
        """按深度分组节点 - 支持同级节点批量处理"""
        depth_groups = {}
        self._collect_by_depth(kg_data["root"], depth_groups)
        return depth_groups
    
    def _collect_by_depth(self, node: Dict[str, Any], groups: Dict):
        """按深度收集节点"""
        depth = node["depth"]
        if depth not in groups:
            groups[depth] = []
        groups[depth].append(node)
        
        if "children" in node:
            for child in node["children"]:
                self._collect_by_depth(child, groups)
    
    # 异步计算方法（保持兼容性）
    async def get_variable_method(self, node: Dict[str, Any]) -> str:
        """获取变量值的方法"""
        await asyncio.sleep(0.01)
        return node["expression"]
    
    async def batch_process_same_level(self, nodes: List[Dict[str, Any]], process_type: str) -> List[str]:
        """批量处理同级节点 - 核心业务功能"""
        results = []
        
        if process_type == "llm_analysis":
            # LLM语义分析
            for node in nodes:
                await asyncio.sleep(0.02)  # 模拟LLM调用
                result = f"LLM分析({node['expression']})"
                results.append(result)
        
        elif process_type == "web_search":
            # 网络搜索
            for node in nodes:
                await asyncio.sleep(0.01)  # 模拟网络请求
                result = f"搜索结果({node['expression']})"
                results.append(result)
        
        elif process_type == "financial_analysis":
            # 财务数据分析
            for node in nodes:
                await asyncio.sleep(0.015)  # 模拟数据库查询
                result = f"财务数据({node['expression']})"
                results.append(result)
        
        return results
    
    async def aggregate_from_leaves(self, kg_data: Dict[str, Any]) -> str:
        """从叶子节点开始向上聚合 - 单层触发机制"""
        result_cache = {}
        batch_counter = 0
        log_lines = []
        execution_flow = []  # 新增：记录执行流程用于可视化
        
        # 持续检查并处理可聚合的层级，直到完成
        while True:
            # 查找当前可以聚合的节点（所有兄弟节点都没有子节点或已完成）
            ready_nodes = self._find_ready_for_aggregation(kg_data, result_cache)
            
            if not ready_nodes:
                break  # 没有更多可聚合的节点
            
            batch_counter += 1
            
            # 记录当前批次并发处理的任务节点
            current_batch_nodes = [f"[{','.join(map(str, node['KG_ID']))}]:{node['expression']}" for node in ready_nodes]
            
            # 新增：记录执行流程信息
            batch_info = {
                'batch_id': batch_counter,
                'nodes': [],
                'operations': []
            }
            
            for node in ready_nodes:
                node_info = {
                    'id': ','.join(map(str, node['KG_ID'])),
                    'expression': node['expression'],
                    'is_variable': node['is_variable'],
                    'operator': node.get('operator', ''),
                    'operation_type': self._get_operation_type(node)
                }
                batch_info['nodes'].append(node_info)
            
            execution_flow.append(batch_info)
            
            # 批量处理当前层的节点
            await self.batch_process_same_level(ready_nodes, "llm_analysis")
            
            '''
            user: 这里的方法等候审查，估计需要改进为可扩展的方法
            # 未来的场景可能是：
            ## get_variable_method: 对应上网搜索某个特定含义的概念，比如 get_variable_method('PPO算法')
            所以该步骤要求：
            1. node["is_variable"] 的生成逻辑最好是该节点不再含有括号和加减法以外的 operator
            例如：
            节点 (A) 或 +A 可以作为get_variable_method的输入（他们对应的operator分别是'()'和'+')
            get_variable_method('A+B') 或 get_variable_method('A*B')会抛出异常，提示输入内容为复杂表达式
            2. 未来A,A1,EXDFT,B之类的变量，会对应数据库中的key，get_variable_method方法体内再实现根据输入的key，去数据库获取真正row_data的能力
            
            # 括号
            定义：需要最优先执行的任务
            案例：
            - 概念解析：比如 ("金刚石半导体"），此时大语言模型可能不知道这个概念的确切含义，需要交给搜索引擎或其他网站（豆包、元宝、Gemini、GPT、Claude）初步回答，界定概念范畴

            # 幂运算（企业层面无法控制的全行业影响因素）
            定义：宏观市场、行业赛道
            案例：
            - 行业爆发： AI推理的井喷，带来了所有关联赛道的收入剧增
            - 行业滑坡： 房地产行业突然崩盘、未来48个月AI应用赛道有崩盘风险
            - 政策影响： 美国对中国AI企业DeepSeek封杀

            # 乘除法（未来：企业通过努力可以改变的市场份额和技术领先）
            定义：来自竞争（蓝海，或加剧竞争）、其他技术路线的挑战
            案例：
            - 乘法-公司市场份额优势：OpenAI处于行业头部地位、谷歌搜索业务占全球90%、高通SoC芯片遥遥领先、Claude正在研发新一代基石模型提升编程能力
            - 除法-巨大挑战：谷歌面临业务拆分、高通失去苹果客户、英伟达推理芯片将逐步被TPU取代
            注意：在我们项目中，乘除法多来自于网络搜索节点的搜素结论（乘法表示支持前一个变量观点，除法表示搜索结论与前一个变量观点相悖）

            # 加减法（现在和过去的企业事实）
            定义：（往往来自投研报告目录或其他场景通用模板）公司介绍、管理团队、核心产品、技术优势、TAM、财务表现、投资风险
            案例：
            - 减法-公司过去12个月收入下降、Intel产品被市场淘汰
            - 加法-PE历史最低点、ARR创新高、个别论坛部分网友的负面报导（特斯拉撞人）
            注意：在我们项目中，加减法多来自于大语言模型的猜测观点（LLM必须在每轮对形成结论的节点给出3个正面和反面意见）

            # 变量：基本观点
            每个变量，都需要通过 搜索-验证 来实现，验证是验证权威性（不能来自不可靠的网站，除非是负面观点）

            *注意
            乘除法与加减法对应的位置不是特别严格，比如：
            谷歌搜索业务占全球90% * 谷歌Gemini模型带来新的AI搜索产品 => 预测搜索业务收入进一步上升
            往往反过来也成立（预测结果的主语有时不太一样）
            谷歌Gemini模型带来新的AI搜索产品 * 谷歌搜索业务占全球90% => 预测AI业务收入进一步上升
            
            '''
            
            # 处理当前批次的每个节点
            parent_nodes = set()
            for node in ready_nodes:
                if node["is_variable"]:
                    result = await self.get_variable_method(node)
                    result_cache[str(node["KG_ID"])] = result
                    # 记录父节点ID（去掉最后一个元素）
                    if len(node["KG_ID"]) > 1:
                        parent_id = node["KG_ID"][:-1]
                        parent_nodes.add(str(parent_id))
                elif "children" in node:
                    result = await self._process_multi_node(node, result_cache)
                    result_cache[str(node["KG_ID"])] = result
                    # 记录父节点ID（去掉最后一个元素）
                    if len(node["KG_ID"]) > 1:
                        parent_id = node["KG_ID"][:-1]
                        parent_nodes.add(str(parent_id))
            
            # 记录本批次日志
            parent_ids_str = ', '.join(sorted(parent_nodes)) if parent_nodes else ""
            log_lines.append(f"批次{batch_counter}:")
            log_lines.append("[")
            for i, node in enumerate(current_batch_nodes):
                log_lines.append(f"  {node}")
            log_lines.append("]")
            log_lines.append("聚合后的父节点ID:")
            log_lines.append(f"[{parent_ids_str}]")
            log_lines.append("")  # 空行分隔
        
        # 输出并发任务执行日志
        import os
        os.makedirs('.cache', exist_ok=True)
        with open('.cache/output.log', 'w', encoding='utf-8') as f:
            f.write('# 并发任务执行批次日志\n')
            f.write('# 格式：批次任务顺序编号, [当前批次并发处理的任务节点], [聚合后的父节点ID]\n\n')
            for line in log_lines:
                f.write(line + '\n')
        
        # 新增：生成执行顺序可视化文件
        self._generate_execution_flow_visualization(execution_flow)
        
        if self.debug_mode:
            print(f"并发任务执行日志已输出到: .cache/output.log")
            print(f"执行顺序可视化已输出到: .cache/execution_flow.mmd")
            print(f"执行顺序详细信息已输出到: .cache/execution_flow.json")
            print(f"文本流程图已输出到: .cache/execution_flow.txt")
        
        root_id = str(kg_data["root"]["KG_ID"])
        return result_cache.get(root_id, "incomplete")
    
    def _find_ready_for_aggregation(self, kg_data: Dict[str, Any], result_cache: Dict[str, str]) -> List[Dict[str, Any]]:
        """查找当前可以聚合的节点（所有兄弟节点都没有子节点或已完成）"""
        ready_nodes = []
        
        def traverse_nodes(nodes: List[Dict[str, Any]], parent_path: List[int] = []):
            for node in nodes:
                node_id = str(node["KG_ID"])
                
                # 如果节点已经处理过，跳过
                if node_id in result_cache:
                    continue
                
                # 检查是否为叶子节点（变量节点）
                if node["is_variable"]:
                    ready_nodes.append(node)
                    continue
                
                # 检查是否为可聚合的非叶子节点
                if "children" in node and self._children_completed(node, result_cache):
                    ready_nodes.append(node)
                    continue
                
                # 递归处理子节点
                if "children" in node:
                    traverse_nodes(node["children"], node["KG_ID"])
        
        # 先遍历所有节点
        if isinstance(kg_data, dict) and "root" in kg_data:
            root_node = kg_data["root"]
            
            # 遍历根节点的子节点
            if "children" in root_node:
                traverse_nodes(root_node["children"])
            
            # 最后检查根节点本身
            root_id = str(root_node["KG_ID"])
            if root_id not in result_cache and self._children_completed(root_node, result_cache):
                ready_nodes.append(root_node)
        elif isinstance(kg_data, dict) and "children" in kg_data:
            traverse_nodes(kg_data["children"])
        elif isinstance(kg_data, list):
            traverse_nodes(kg_data)
        
        return ready_nodes
    
    def _get_operation_type(self, node: Dict[str, Any]) -> str:
        """获取节点的操作类型，用于可视化"""
        if node["is_variable"]:
            return "变量获取"
        
        operator = node.get("operator", "")
        if operator == "()":
            return "括号运算(概念解析)"
        elif operator in ["+", "-"]:
            return "加减法(企业事实)"
        elif operator in ["*", "/"]:
            return "乘除法(市场竞争)"
        elif operator == "^":
            return "幂运算(行业因素)"
        else:
            return "默认运算"
    
    def _generate_execution_flow_visualization(self, execution_flow: List[Dict]) -> None:
        """生成执行顺序可视化文件"""
        import os
        import json
        
        # 输出详细的执行流程JSON
        with open('.cache/execution_flow.json', 'w', encoding='utf-8') as f:
            json.dump(execution_flow, f, ensure_ascii=False, indent=2)
        
        # 生成Mermaid流程图
        mermaid_lines = []
        mermaid_lines.append("graph TD")
        mermaid_lines.append("    %% 知识图谱执行顺序可视化")
        mermaid_lines.append("")
        
        # 添加开始节点
        mermaid_lines.append("    START([开始解析])")
        
        prev_batch_nodes = ["START"]
        
        for batch in execution_flow:
            batch_id = batch['batch_id']
            nodes = batch['nodes']
            
            # 为每个批次创建一个批次标识节点
            batch_node = f"BATCH{batch_id}"
            mermaid_lines.append(f"    {batch_node}[批次{batch_id}]")
            
            # 连接上一批次到当前批次
            for prev_node in prev_batch_nodes:
                mermaid_lines.append(f"    {prev_node} --> {batch_node}")
            
            # 为当前批次的每个节点创建节点
            current_batch_nodes = []
            for i, node in enumerate(nodes):
                node_id = f"N{batch_id}_{i}"
                node_label = f"{node['expression']}\\n({node['operation_type']})"
                
                # 根据操作类型设置不同的样式
                if node['is_variable']:
                    mermaid_lines.append(f"    {node_id}[{node_label}]")
                    mermaid_lines.append(f"    {node_id} --> {node_id}_result{{获取: {node['expression']}}}")
                    current_batch_nodes.append(f"{node_id}_result")
                else:
                    mermaid_lines.append(f"    {node_id}[{node_label}]")
                    mermaid_lines.append(f"    {node_id} --> {node_id}_result{{聚合: {node['expression']}}}")
                    current_batch_nodes.append(f"{node_id}_result")
                
                # 连接批次节点到具体操作节点
                mermaid_lines.append(f"    {batch_node} --> {node_id}")
            
            prev_batch_nodes = current_batch_nodes
        
        # 添加结束节点
        mermaid_lines.append("    END([完成聚合])")
        for node in prev_batch_nodes:
            mermaid_lines.append(f"    {node} --> END")
        
        # 添加样式定义
        mermaid_lines.append("")
        mermaid_lines.append("    %% 样式定义")
        mermaid_lines.append("    classDef batchClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px")
        mermaid_lines.append("    classDef variableClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px")
        mermaid_lines.append("    classDef operationClass fill:#fff3e0,stroke:#e65100,stroke-width:2px")
        mermaid_lines.append("    classDef resultClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px")
        
        # 输出Mermaid文件
        with open('.cache/execution_flow.mmd', 'w', encoding='utf-8') as f:
            f.write('\n'.join(mermaid_lines))
        
        # 生成简化的文本流程图
        self._generate_simple_text_flow(execution_flow)
    
    def _generate_simple_text_flow(self, execution_flow: List[Dict]) -> None:
        """生成简化的文本流程图，更易读的格式"""
        text_lines = []
        text_lines.append("# 知识图谱执行顺序流程图")
        text_lines.append("")
        text_lines.append("```")
        text_lines.append("┌─────────────────┐")
        text_lines.append("│   开始解析      │")
        text_lines.append("└─────────────────┘")
        text_lines.append("         │")
        text_lines.append("         ▼")
        
        for i, batch in enumerate(execution_flow):
            batch_id = batch['batch_id']
            nodes = batch['nodes']
            
            # 批次标题
            text_lines.append(f"┌─────────────────┐")
            text_lines.append(f"│   批次 {batch_id}        │")
            text_lines.append(f"└─────────────────┘")
            
            # 并发执行的节点
            if len(nodes) > 1:
                text_lines.append("         │")
                text_lines.append("    ┌────┴────┐")
                
                # 绘制并发分支
                for j, node in enumerate(nodes):
                    if j == 0:
                        text_lines.append(f"    ▼         ▼")
                    
                    operation_symbol = self._get_operation_symbol(node)
                    expr_short = node['expression'][:8] + '...' if len(node['expression']) > 8 else node['expression']
                    
                    if j % 2 == 0:  # 左侧
                        text_lines.append(f"┌─────────┐ ┌─────────┐")
                        text_lines.append(f"│{operation_symbol} {expr_short:<7}│ │         │")
                        text_lines.append(f"└─────────┘ └─────────┘")
                    else:  # 右侧
                        # 修改上一行
                        text_lines[-2] = text_lines[-2].replace("│         │", f"│{operation_symbol} {expr_short:<7}│")
                
                # 合并回主流程
                text_lines.append("    │         │")
                text_lines.append("    └────┬────┘")
                text_lines.append("         │")
            else:
                # 单个节点
                node = nodes[0]
                operation_symbol = self._get_operation_symbol(node)
                expr_short = node['expression'][:10] + '...' if len(node['expression']) > 10 else node['expression']
                
                text_lines.append("         │")
                text_lines.append("         ▼")
                text_lines.append(f"┌─────────────────┐")
                text_lines.append(f"│{operation_symbol} {expr_short:<13}│")
                text_lines.append(f"└─────────────────┘")
            
            # 如果不是最后一个批次，添加连接线
            if i < len(execution_flow) - 1:
                text_lines.append("         │")
                text_lines.append("         ▼")
        
        # 结束
        text_lines.append("         │")
        text_lines.append("         ▼")
        text_lines.append("┌─────────────────┐")
        text_lines.append("│   完成聚合      │")
        text_lines.append("└─────────────────┘")
        text_lines.append("```")
        
        # 添加图例说明
        text_lines.append("")
        text_lines.append("## 图例说明")
        text_lines.append("")
        text_lines.append("- 🔤 变量获取：从数据库或搜索引擎获取基础数据")
        text_lines.append("- 📦 括号运算：概念解析，需要LLM理解复杂概念")
        text_lines.append("- ➕ 加减法：企业历史事实和现状分析")
        text_lines.append("- ✖️ 乘除法：市场竞争和技术领先性分析")
        text_lines.append("- 🔺 幂运算：行业宏观因素和政策影响")
        text_lines.append("")
        
        # 添加详细的批次信息
        text_lines.append("## 详细执行信息")
        text_lines.append("")
        for batch in execution_flow:
            text_lines.append(f"### 批次 {batch['batch_id']}")
            text_lines.append("")
            for node in batch['nodes']:
                text_lines.append(f"- **{node['expression']}** ({node['operation_type']})")
                if node['operator']:
                    text_lines.append(f"  - 操作符: `{node['operator']}`")
                text_lines.append(f"  - 节点ID: `{node['id']}`")
                text_lines.append("")
        
        # 输出文本流程图
        with open('.cache/execution_flow.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_lines))
    
    def _get_operation_symbol(self, node: Dict[str, Any]) -> str:
        """获取操作的符号表示"""
        if node['is_variable']:
            return "🔤"
        
        operator = node.get('operator', '')
        if operator == "()":
            return "📦"
        elif operator in ["+", "-"]:
            return "➕"
        elif operator in ["*", "/"]:
            return "✖️"
        elif operator == "^":
            return "🔺"
        else:
            return "⚙️"
    
    def _children_completed(self, node: Dict[str, Any], result_cache: Dict[str, str]) -> bool:
        """检查节点的所有子节点是否都已完成计算"""
        if "children" not in node:
            return True
        
        for child in node["children"]:
            child_id = str(child["KG_ID"])
            if child_id not in result_cache:
                return False
        return True
    
    async def _process_multi_node(self, node: Dict[str, Any], cache: Dict):
        """处理多叉树节点 - 调度到具体的原子操作方法"""
        children = node["children"]
        
        if len(children) == 1:
            # 单子节点处理
            child_result = cache[str(children[0]['KG_ID'])]
            if children[0].get("operator") == "()":
                return await self._process_parentheses_operation(node, cache)
            return child_result
        
        # 多子节点：根据运算符类型调度到对应的原子操作方法
        operators = [children[i]["operator"] for i in range(1, len(children))]
        
        # 检查运算符类型并调度
        if any(op in ['+', '-'] for op in operators):
            return await self._process_addition_subtraction_operation(node, cache)
        elif any(op in ['*', '/'] for op in operators):
            return await self._process_multiplication_division_operation(node, cache)
        elif any(op == '^' for op in operators):
            return await self._process_power_operation(node, cache)
        else:
            # 默认处理
            return await self._process_default_operation(node, cache)
    
    async def _process_parentheses_operation(self, node: Dict[str, Any], cache: Dict) -> str:
        """括号运算原子操作 - 最优先执行的任务
        
        定义：需要最优先执行的任务
        案例：概念解析，比如 ("金刚石半导体")，需要交给搜索引擎或LLM初步回答
        
        当前版本：简单字符串包装
        未来扩展：可添加概念解析、搜索引擎查询等功能
        """
        print("[DEBUG] 执行括号运算方法: _process_parentheses_operation")
        children = node["children"]
        child_result = cache[str(children[0]['KG_ID'])]
        
        # 当前版本：简单字符串相加
        result = f"({child_result})"
        
        # TODO: 未来扩展点
        # - 概念解析：调用搜索引擎或LLM解释概念
        # - 权威性验证：检查信息来源可靠性
        
        return result
    
    async def _process_addition_subtraction_operation(self, node: Dict[str, Any], cache: Dict) -> str:
        """加减法运算原子操作 - 现在和过去的企业事实
        
        定义：公司介绍、管理团队、核心产品、技术优势、TAM、财务表现、投资风险
        案例：
        - 加法：PE历史最低点、ARR创新高
        - 减法：公司过去12个月收入下降、Intel产品被市场淘汰
        
        当前版本：简单字符串相加
        未来扩展：LLM必须给出3个正面和反面意见
        """
        print("[DEBUG] 执行加减法运算方法: _process_addition_subtraction_operation")
        children = node["children"]
        result = cache[str(children[0]["KG_ID"])]
        
        for i in range(1, len(children)):
            child_val = cache[str(children[i]["KG_ID"])]
            operator = children[i]["operator"]
            
            # 当前版本：简单字符串相加
            result = f"{result}{operator}{child_val}"
            
            # TODO: 未来扩展点
            # - 加法：LLM分析正面因素，生成支持观点
            # - 减法：LLM分析负面因素，生成风险评估
            # - 权威性验证：确保信息来源可靠
        
        return result
    
    async def _process_multiplication_division_operation(self, node: Dict[str, Any], cache: Dict) -> str:
        """乘除法运算原子操作 - 企业通过努力可以改变的市场份额和技术领先
        
        定义：来自竞争（蓝海，或加剧竞争）、其他技术路线的挑战
        案例：
        - 乘法：OpenAI处于行业头部地位、谷歌搜索业务占全球90%
        - 除法：谷歌面临业务拆分、高通失去苹果客户
        
        当前版本：简单字符串相加
        未来扩展：网络搜索节点的搜索结论（乘法表示支持，除法表示相悖）
        """
        print("[DEBUG] 执行乘除法运算方法: _process_multiplication_division_operation")
        children = node["children"]
        result = cache[str(children[0]["KG_ID"])]
        
        for i in range(1, len(children)):
            child_val = cache[str(children[i]["KG_ID"])]
            operator = children[i]["operator"]
            
            # 当前版本：简单字符串相加
            result = f"{result}{operator}{child_val}"
            
            # TODO: 未来扩展点
            # - 乘法：网络搜索支持前一个变量观点的证据
            # - 除法：网络搜索与前一个变量观点相悖的证据
            # - 竞争分析：市场份额、技术领先性评估
        
        return result
    
    async def _process_power_operation(self, node: Dict[str, Any], cache: Dict) -> str:
        """幂运算原子操作 - 企业层面无法控制的全行业影响因素
        
        定义：宏观市场、行业赛道
        案例：
        - 行业爆发：AI推理的井喷，带来所有关联赛道的收入剧增
        - 行业滑坡：房地产行业突然崩盘
        - 政策影响：美国对中国AI企业DeepSeek封杀
        
        当前版本：简单字符串相加
        未来扩展：宏观经济、政策影响、行业趋势分析
        """
        print("[DEBUG] 执行幂运算方法: _process_power_operation")
        children = node["children"]
        result = cache[str(children[0]["KG_ID"])]
        
        for i in range(1, len(children)):
            child_val = cache[str(children[i]["KG_ID"])]
            operator = children[i]["operator"]
            
            # 当前版本：简单字符串相加
            result = f"{result}{operator}{child_val}"
            
            # TODO: 未来扩展点
            # - 宏观经济分析：行业整体趋势评估
            # - 政策影响分析：政府政策对行业的影响
            # - 不可控因素：企业无法改变的外部环境
        
        return result
    
    async def _process_default_operation(self, node: Dict[str, Any], cache: Dict) -> str:
        """默认运算原子操作 - 处理其他未分类的运算符
        
        当前版本：简单字符串相加
        未来扩展：根据具体业务需求添加新的运算符处理逻辑
        """
        print("[DEBUG] 执行默认运算方法: _process_default_operation")
        children = node["children"]
        result = cache[str(children[0]["KG_ID"])]
        
        for i in range(1, len(children)):
            child_val = cache[str(children[i]["KG_ID"])]
            operator = children[i]["operator"]
            
            # 当前版本：简单字符串相加
            if operator == "()":
                result = f"({child_val})"
            else:
                result = f"{result}{operator}{child_val}"
        
        return result
    
    def to_markdown_tree(self, kg_data: Dict[str, Any]) -> str:
        """将知识图谱转换为 Markdown 树结构"""
        lines = ["# 知识图谱结构示例 (多叉树)\n\n```json\n"]
        self._add_markdown_children(kg_data["root"], lines, "", True)
        lines.append("```\n")
        return ''.join(lines)
    
    def _add_markdown_children(self, node: Dict[str, Any], lines: List[str], prefix: str, is_last: bool = True):
        """添加 Markdown 子节点"""
        kg_id = node["KG_ID"]
        expr = node["expression"]
        op = node["operator"]
        is_var = node["is_variable"]
        depth = node["depth"]
        priority = node["priority"]
        
        # 格式化节点ID
        id_str = "[" + ",".join(map(str, kg_id)) + "]"
        
        # 构建完整的JSON字段信息
        json_info = f'{{"KG_ID": {kg_id}, "expression": "{expr}", "is_variable": {str(is_var).lower()}, "operator": "{op}", "depth": {depth}, "priority": {priority}}}'
        
        # 根节点不需要分支符号
        if prefix == "":
            lines.append(f"{id_str}: {json_info}\n")
        else:
            branch = "└── " if is_last else "├── "
            lines.append(f"{prefix}{branch}{id_str}: {json_info}\n")
        
        if "children" in node and node["children"]:
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node["children"]):
                is_child_last = i == len(node["children"]) - 1
                self._add_markdown_children(child, lines, child_prefix, is_child_last)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='知识图谱解析器 - 多叉树结构')
    parser.add_argument('expression', nargs='?', default='A1^2+((B1*2-C1)/D1^(E+F)+G)*H-1', 
                       help='要解析的数学表达式')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    print(f"使用表达式: {args.expression}")
    
    # 创建解析器
    kg_parser = KnowledgeGraphParser(debug_mode=args.debug)
    
    # 解析为知识图谱
    kg_data = kg_parser.parse_to_kg(args.expression)
    
    # 输出到文件
    import os
    os.makedirs('.cache', exist_ok=True)
    with open('.cache/output.json', 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)
    print("知识图谱已输出到: .cache/output.json")
    
    # 输出 Markdown 树结构
    markdown_tree = kg_parser.to_markdown_tree(kg_data)
    with open('.cache/output.md', 'w', encoding='utf-8') as f:
        f.write(markdown_tree)
    print("Markdown 树结构已输出到: .cache/output.md")
    
    print("\n=== 执行顺序可视化文件 ===")
    print("Mermaid流程图: .cache/execution_flow.mmd")
    print("文本流程图: .cache/execution_flow.txt")
    print("详细执行信息: .cache/execution_flow.json")
    
    # 演示同级节点批量处理
    if args.debug:
        print("\n=== 同级节点批量处理演示 ===")
        depth_groups = kg_parser.get_nodes_by_depth(kg_data)
        for depth, nodes in depth_groups.items():
            if depth > 0 and len(nodes) > 1:  # 有多个同级节点
                print(f"\n深度 {depth} 的同级节点数量: {len(nodes)}")
                for node in nodes:
                    print(f"  - {node['expression']} (优先级: {node['priority']})")
    
    # 异步聚合计算
    async def run_aggregation():
        result = await kg_parser.aggregate_from_leaves(kg_data)
        print(f"聚合结果: {result}")
    
    asyncio.run(run_aggregation())

if __name__ == "__main__":
    main()