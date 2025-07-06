# 复杂表达式解析器 (JavaScript版)

一个基于知识图谱的复杂数学表达式解析与重建系统，支持多层嵌套括号、多种运算符优先级处理。

## 核心特性

- **多叉树结构**: 将复杂表达式解析为层次化的知识图谱
- **从叶子节点重建**: 实现从底层变量开始的递归表达式重建
- **运算符优先级**: 支持加减法(20)、乘除法(40)、幂运算(60)的优先级处理
- **括号嵌套**: 完整支持多层括号嵌套和概念解析

## 算法原理

### 1. 解析算法 (parseToKG)

解析器采用**递归下降解析**算法，将表达式转换为多叉树知识图谱：

```
表达式: A1^2+((B1*2-C1)/D1^(E+F)+G)*H-1

解析流程:
1. 词法分析 → Token序列
2. 语法分析 → AST抽象语法树
3. 知识图谱转换 → 多叉树结构
```

### 2. 重建算法 (aggregateFromLeaves)

重建算法采用**从叶子节点开始的递归聚合**策略：

```
重建流程:
1. 识别叶子节点(变量节点)
2. 按运算符类型调度处理方法
3. 递归向上聚合父节点
4. 最终重建完整表达式
```

## 多叉树结构

### 节点类型

- **变量节点**: `is_variable: true`，表示基础变量或数字
- **运算节点**: `is_variable: false`，包含子节点和运算符信息
- **括号节点**: `operator: "()"`，处理括号概念解析

### 节点属性

```javascript
{
  "KG_ID": [0, 1, 0],        // 唯一标识路径
  "expression": "A1^2",      // 节点表达式
  "is_variable": false,      // 是否为变量节点
  "operator": "^",           // 运算符
  "depth": 2,                // 树深度
  "priority": 3,             // 优先级
  "children": [...]          // 子节点数组
}
```

## 实际案例

### 测试表达式
```
A1^2+((B1*2-C1)/D1^(E+F)+G)*H-1
```

### 解析结果

根据 <mcfile name="compare.log" path=".cache/compare.log"></mcfile> 的测试结果：

- **原始表达式**: `A1^2+((B1*2-C1)/D1^(E+F)+G)*H-1`
- **重建表达式**: `A1^2+((B1*2-C1)/D1^(E+F)+G)*H-1`
- **表达式长度**: 原始=31, 重建=31
- **表达式一致**: ✅ true

### 知识图谱结构示例

以根节点为例，完整的多叉树结构包含：

```json
{
  "KG_ID": [0],
  "expression": "A1^2+((B1*2-C1)/D1^(E+F)+G)*H-1",
  "is_variable": false,
  "operator": "",
  "depth": 0,
  "priority": 1,
  "children": [
    {
      "KG_ID": [0, 0],
      "expression": "A1^2",
      "operator": "",
      "priority": 3,
      "children": [
        {"expression": "A1", "is_variable": true},
        {"expression": "2", "operator": "^", "is_variable": true}
      ]
    },
    {
      "KG_ID": [0, 1],
      "expression": "((B1*2-C1)/D1^(E+F)+G)*H",
      "operator": "+",
      "priority": 2,
      "children": [...]
    }
  ]
}
```

### 处理方法调度

根据运算符类型，系统自动调度到相应的处理方法：

- **括号运算**: `processParenthesesOperation` - 概念解析
- **加减法**: `processAdditionSubtractionOperation` - 企业事实
- **乘除法**: `processMultiplicationDivisionOperation` - 市场竞争
- **幂运算**: `processPowerOperation` - 行业因素
- **默认运算**: `processDefaultOperation` - 通用处理

## API 参考

### KnowledgeGraphParser 类

#### 构造函数
```javascript
new KnowledgeGraphParser(debugMode = false)
```
- `debugMode`: 布尔值，是否开启调试模式

#### 核心方法

##### parseToKG(expression)
将表达式解析为知识图谱结构
- **参数**: `expression` (string) - 要解析的数学表达式
- **返回**: `{root: KGNode}` - 包含根节点的知识图谱对象
- **功能**: 执行词法分析、语法分析、AST转换为知识图谱

##### rebuildExpressionFromKG(kgNode)
从知识图谱节点重建表达式
- **参数**: `kgNode` (object) - 知识图谱节点
- **返回**: `string` - 重建的表达式字符串
- **功能**: 从叶子节点开始递归重建表达式

##### async aggregateFromLeaves(kgData)
异步聚合处理，从叶子节点开始批量处理
- **参数**: `kgData` (object) - 包含根节点的知识图谱数据
- **返回**: `Promise<string>` - 最终聚合结果
- **功能**: 模拟并发任务执行，按批次处理节点，生成执行日志

#### 辅助方法

##### ensureCacheDir()
确保缓存目录存在

##### saveASTStructure(astRoot, filename, originalExpr)
保存AST结构到文件
- **参数**: 
  - `astRoot` - AST根节点
  - `filename` - 输出文件名
  - `originalExpr` - 原始表达式

##### async batchProcessSameLevel(nodes, processType)
批量处理同级节点
- **参数**:
  - `nodes` - 节点数组
  - `processType` - 处理类型 ("llm_analysis", "web_search", "financial_analysis")
- **返回**: `Promise<Array>` - 处理结果数组

##### async getVariableMethod(node)
异步获取变量值
- **参数**: `node` - 变量节点
- **返回**: `Promise<string>` - 变量表达式

#### 运算处理方法

##### async processParenthesesOperation(node, cache)
处理括号运算

##### async processAdditionSubtractionOperation(node, cache)
处理加减法运算

##### async processMultiplicationDivisionOperation(node, cache)
处理乘除法运算

##### async processPowerOperation(node, cache)
处理幂运算

##### async processDefaultOperation(node, cache)
处理默认运算

### 使用示例

```javascript
import { KnowledgeGraphParser } from './KnowledgeGraph_Expression_Parser.js';

// 创建解析器实例
const parser = new KnowledgeGraphParser(true); // true开启调试模式

// 解析表达式为知识图谱
const expression = "A1^2+((B1*2-C1)/D1^(E+F)+G)*H-1";
const kgData = parser.parseToKG(expression);

// 从知识图谱重建表达式
const rebuiltExpression = parser.rebuildExpressionFromKG(kgData.root);

// 异步聚合处理（从叶子节点开始）
const aggregatedResult = await parser.aggregateFromLeaves(kgData);

console.log('原始表达式:', expression);
console.log('重建表达式:', rebuiltExpression);
console.log('聚合结果:', aggregatedResult);
```

### 运行测试

```bash
node comprehensive_test.js
```

测试程序将生成以下输出文件：
- <mcfile name="compare.log" path=".cache/compare.log"></mcfile>: 表达式对比结果
- <mcfile name="knowledge_graph.ast" path=".cache/knowledge_graph.ast"></mcfile>: 知识图谱AST结构信息
- <mcfile name="process.log" path=".cache/process.log"></mcfile>: 表达式解析处理日志
- <mcfile name="output.log" path=".cache/output.log"></mcfile>: 并发任务执行批次日志（由aggregateFromLeaves方法生成）

## 技术架构

### 核心组件

1. **词法分析器 (Lexer)**: 将表达式转换为Token序列
2. **语法分析器 (Parser)**: 构建抽象语法树
3. **知识图谱解析器 (KnowledgeGraphParser)**: 转换为多叉树结构
4. **聚合处理器**: 从叶子节点重建表达式

### 运算符优先级表

| 运算符 | 优先级 | 类型 |
|--------|--------|------|
| `+`, `-` | 20 | 加减法 |
| `*`, `/` | 40 | 乘除法 |
| `^` | 60 | 幂运算 |
| `()` | 100 | 括号 |

## 项目结构

```
复杂表达式解析JS/
├── KnowledgeGraph_Expression_Parser.js  # 主解析器
├── comprehensive_test.js                # 综合测试程序
├── .cache/                             # 测试输出目录
│   ├── compare.log                     # 对比结果
│   ├── output.ast                      # AST结构
│   └── output.log                      # 处理日志
└── README.md                           # 项目文档
```

## 特色功能

### 1. 深度嵌套支持

系统支持任意深度的括号嵌套，如测试案例中的 `((B1*2-C1)/D1^(E+F)+G)` 结构。

### 2. 智能优先级处理

根据数学运算规则，自动处理运算符优先级，确保表达式语义正确。

### 3. 完整性验证

通过对比原始表达式与重建表达式，验证解析和重建算法的正确性。

## 开发说明

本项目采用ES6模块化开发，支持现代JavaScript环境。核心算法基于递归下降解析和多叉树数据结构，确保了高效的表达式处理能力。

---

*基于知识图谱的表达式解析技术，为复杂数学表达式处理提供可靠解决方案。*