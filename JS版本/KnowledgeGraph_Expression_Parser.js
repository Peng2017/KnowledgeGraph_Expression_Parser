import * as fs from 'fs';
import * as path from 'path';

// Token类型枚举
const TokenType = {
    LITERAL: "LITERAL",      // 数字
    IDENTIFIER: "IDENTIFIER", // 变量名
    OPERATOR: "OPERATOR",     // 运算符
    LPAREN: "LPAREN",         // 左括号
    RPAREN: "RPAREN",         // 右括号
    EOF: "EOF"               // 结束符
};

// Token类
class Token {
    constructor(type, value, position = 0) {
        this.type = type;
        this.value = value;
        this.position = position;
    }
    
    toString() {
        return `Token(${this.type}, ${this.value})`;
    }
}

// 词法分析器
class Lexer {
    constructor(text) {
        this.text = text.replace(/\s/g, ''); // 移除空格
        this.position = 0;
        this.currentChar = this.text.length > 0 ? this.text[0] : null;
    }
    
    advance() {
        this.position++;
        if (this.position >= this.text.length) {
            this.currentChar = null;
        } else {
            this.currentChar = this.text[this.position];
        }
    }
    
    readNumber() {
        let result = '';
        while (this.currentChar !== null && (this.currentChar.match(/\d/) || this.currentChar === '.')) {
            result += this.currentChar;
            this.advance();
        }
        return result;
    }
    
    readIdentifier() {
        let result = '';
        while (this.currentChar !== null && this.currentChar.match(/[a-zA-Z0-9_]/)) {
            result += this.currentChar;
            this.advance();
        }
        return result;
    }
    
    tokenize() {
        const tokens = [];
        
        while (this.currentChar !== null) {
            if (this.currentChar.match(/\d/)) {
                const number = this.readNumber();
                tokens.push(new Token(TokenType.LITERAL, number, this.position));
            }
            else if (this.currentChar.match(/[a-zA-Z]/)) {
                const identifier = this.readIdentifier();
                tokens.push(new Token(TokenType.IDENTIFIER, identifier, this.position));
            }
            else if (this.currentChar === '(') {
                tokens.push(new Token(TokenType.LPAREN, '(', this.position));
                this.advance();
            }
            else if (this.currentChar === ')') {
                tokens.push(new Token(TokenType.RPAREN, ')', this.position));
                this.advance();
            }
            else if (['+', '-', '*', '/', '^'].includes(this.currentChar)) {
                tokens.push(new Token(TokenType.OPERATOR, this.currentChar, this.position));
                this.advance();
            }
            else {
                this.advance();
            }
        }
        
        tokens.push(new Token(TokenType.EOF, '', this.position));
        return tokens;
    }
}

// AST节点基类
class ASTNode {}

// 数字节点
class NumberNode extends ASTNode {
    constructor(value) {
        super();
        this.value = value;
    }
    
    toString() {
        return `NumberNode(${this.value})`;
    }
}

// 变量节点
class VariableNode extends ASTNode {
    constructor(name) {
        super();
        this.name = name;
    }
    
    toString() {
        return `VariableNode(${this.name})`;
    }
}

// 多元运算节点
class MultiOpNode extends ASTNode {
    constructor(operands, operators, priority) {
        super();
        this.operands = operands;
        this.operators = operators;
        this.priority = priority;
    }
    
    toString() {
        return `MultiOpNode(operands=${this.operands.length}, ops=${this.operators}, priority=${this.priority})`;
    }
}

// 括号节点
class ParenthesesNode extends ASTNode {
    constructor(innerExpr) {
        super();
        this.innerExpr = innerExpr;
    }
    
    toString() {
        return `ParenthesesNode(${this.innerExpr})`;
    }
}

// 递归下降解析器
class Parser {
    static PRECEDENCE = {
        '+': 20, '-': 20,
        '*': 40, '/': 40,
        '^': 60
    };
    
    constructor(tokens) {
        this.tokens = tokens;
        this.position = 0;
        this.currentToken = tokens.length > 0 ? tokens[0] : new Token(TokenType.EOF, '');
    }
    
    advance() {
        this.position++;
        if (this.position < this.tokens.length) {
            this.currentToken = this.tokens[this.position];
        } else {
            this.currentToken = new Token(TokenType.EOF, '');
        }
    }
    
    getPrecedence(operator) {
        return Parser.PRECEDENCE[operator] || -1;
    }
    
    parseExpression() {
        return this.parsePrecedenceLevel(0);
    }
    
    parsePrecedenceLevel(minPrecedence) {
        let left = this.parsePrimary();
        
        while (this.currentToken.type === TokenType.OPERATOR && 
               this.getPrecedence(this.currentToken.value) >= minPrecedence) {
            
            const operands = [left];
            const operators = [];
            const currentPrecedence = this.getPrecedence(this.currentToken.value);
            
            while (this.currentToken.type === TokenType.OPERATOR && 
                   this.getPrecedence(this.currentToken.value) === currentPrecedence) {
                
                const op = this.currentToken.value;
                operators.push(op);
                this.advance();
                
                const right = this.parsePrecedenceLevel(currentPrecedence + 1);
                operands.push(right);
            }
            
            if (operands.length === 1) {
                left = operands[0];
            } else {
                left = new MultiOpNode(operands, operators, currentPrecedence);
            }
        }
        
        return left;
    }
    
    parsePrimary() {
        if (this.currentToken.type === TokenType.LITERAL) {
            const value = this.currentToken.value;
            this.advance();
            return new NumberNode(value);
        }
        else if (this.currentToken.type === TokenType.IDENTIFIER) {
            const name = this.currentToken.value;
            this.advance();
            return new VariableNode(name);
        }
        else if (this.currentToken.type === TokenType.LPAREN) {
            this.advance(); // 跳过 '('
            const expr = this.parseExpression();
            if (this.currentToken.type !== TokenType.RPAREN) {
                throw new Error(`Expected ')' but got ${this.currentToken}`);
            }
            this.advance(); // 跳过 ')'
            return new ParenthesesNode(expr);
        }
        else if (this.currentToken.type === TokenType.OPERATOR && this.currentToken.value === '-') {
            // 负号处理
            this.advance();
            const operand = this.parsePrimary();
            return new MultiOpNode([new NumberNode('0'), operand], ['-'], 20);
        }
        else {
            throw new Error(`Unexpected token: ${this.currentToken}`);
        }
    }
}

// 知识图谱解析器
class KnowledgeGraphParser {
    constructor(debugMode = false) {
        this.debugMode = debugMode;
        this.priorityLevels = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3};
    }
    
    parseToKG(expression) {
        // 词法分析
        const lexer = new Lexer(expression);
        const tokens = lexer.tokenize();
        
        // 语法分析
        const parser = new Parser(tokens);
        const astRoot = parser.parseExpression();
        
        // 保存AST结构
        this.ensureCacheDir();
        this.saveASTStructure(astRoot, ".cache/output.ast", expression);
        
        // 转换为知识图谱
        const kgRoot = this.astToKG(astRoot, [0], 0, expression);
        
        return {root: kgRoot};
    }
    
    ensureCacheDir() {
        if (!fs.existsSync('.cache')) {
            fs.mkdirSync('.cache', {recursive: true});
        }
    }
    
    saveASTStructure(astRoot, filename, originalExpr) {
        const content = [];
        content.push("# AST 解析结果 (多叉树)\n");
        content.push(`## 原始表达式: ${originalExpr}\n\n`);
        
        content.push("## AST 树形结构:\n```\n");
        this.formatASTTree(astRoot, content, "");
        content.push("\n```\n\n");
        
        if (this.debugMode) {
            content.push("## 调试信息:\n");
            content.push("### 解析器类型: 递归下降解析器（多叉树结构）\n");
            content.push("### 运算符优先级表:\n");
            content.push("```\n");
            for (const [op, prec] of Object.entries(Parser.PRECEDENCE)) {
                content.push(`${op}: ${prec}\n`);
            }
            content.push("```\n");
        }
        
        fs.writeFileSync(filename, content.join(''), 'utf8');
        
        if (this.debugMode) {
            console.log(`AST 结构已保存到: ${filename}`);
        }
    }
    
    formatASTTree(node, lines, prefix) {
        if (node instanceof NumberNode) {
            lines.push(`NumberNode(${node.value})\n`);
        } else if (node instanceof VariableNode) {
            lines.push(`VariableNode(${node.name})\n`);
        } else if (node instanceof ParenthesesNode) {
            lines.push(`ParenthesesNode()\n`);
            lines.push(`${prefix}└── inner: `);
            this.formatASTTree(node.innerExpr, lines, prefix + "    ");
        } else if (node instanceof MultiOpNode) {
            lines.push(`MultiOpNode(priority=${node.priority}, ops=${node.operators})\n`);
            for (let i = 0; i < node.operands.length; i++) {
                const isLast = i === node.operands.length - 1;
                const branch = isLast ? "└── " : "├── ";
                const nextPrefix = isLast ? "    " : "│   ";
                lines.push(`${prefix}${branch}operand[${i}]: `);
                this.formatASTTree(node.operands[i], lines, prefix + nextPrefix);
            }
        }
    }
    
    astToKG(node, kgId, depth, originalExpr) {
        if (node instanceof NumberNode) {
            return {
                KG_ID: kgId,
                expression: node.value,
                is_variable: true,
                operator: "",
                depth: depth,
                priority: 0
            };
        }
        else if (node instanceof VariableNode) {
            return {
                KG_ID: kgId,
                expression: node.name,
                is_variable: true,
                operator: "",
                depth: depth,
                priority: 0
            };
        }
        else if (node instanceof ParenthesesNode) {
            const childId = [...kgId, 0];
            const child = this.astToKG(node.innerExpr, childId, depth + 1, originalExpr);
            child.operator = "()";
            
            return {
                KG_ID: kgId,
                expression: `(${this.extractNodeExpression(node.innerExpr)})`,
                is_variable: false,
                operator: "()",
                depth: depth,
                priority: 100,
                children: [child]
            };
        }
        else if (node instanceof MultiOpNode) {
            const expression = depth === 0 ? originalExpr : this.buildExpression(node);
            
            const children = [];
            for (let i = 0; i < node.operands.length; i++) {
                const childId = [...kgId, i];
                const childOp = i === 0 ? "" : node.operators[i-1];
                const child = this.astToKG(node.operands[i], childId, depth + 1, originalExpr);
                child.operator = childOp;
                children.push(child);
            }
            
            return {
                KG_ID: kgId,
                expression: expression,
                is_variable: false,
                operator: "",
                depth: depth,
                priority: this.priorityLevels[node.operators[0]] || 0,
                children: children
            };
        }
    }
    
    buildExpression(node) {
        if (node.operands.length === 1) {
            return this.extractNodeExpression(node.operands[0]);
        }
        
        const parts = [this.extractNodeExpression(node.operands[0])];
        for (let i = 0; i < node.operators.length; i++) {
            if (i + 1 < node.operands.length) {
                parts.push(node.operators[i] + this.extractNodeExpression(node.operands[i + 1]));
            }
        }
        
        return parts.join('');
    }
    
    extractNodeExpression(node) {
        if (node instanceof NumberNode) {
            return node.value;
        } else if (node instanceof VariableNode) {
            return node.name;
        } else if (node instanceof ParenthesesNode) {
            return `(${this.extractNodeExpression(node.innerExpr)})`;
        } else if (node instanceof MultiOpNode) {
            return this.buildExpression(node);
        }
        return "unknown";
    }
    
    // 从知识图谱节点重建表达式 - 从叶子节点开始递归重建
    rebuildExpressionFromKG(kgNode) {
        // 如果是变量节点（叶子节点），直接返回表达式
        if (kgNode.is_variable) {
            return kgNode.expression;
        }
        
        // 如果没有子节点，直接返回表达式
        if (!kgNode.children || kgNode.children.length === 0) {
            return kgNode.expression;
        }
        
        // 递归处理子节点，从叶子节点开始重建
        return this._processMultiNode(kgNode);
    }
    
    // 处理多叉树节点 - 模拟Python版本的_process_multi_node逻辑
    _processMultiNode(node) {
        const children = node.children;
        
        if (children.length === 1) {
            // 单子节点处理
            const childResult = this.rebuildExpressionFromKG(children[0]);
            if (children[0].operator === "()") {
                return this._processParenthesesOperation(node);
            }
            return childResult;
        }
        
        // 多子节点：根据运算符类型调度到对应的处理方法
        const operators = [];
        for (let i = 1; i < children.length; i++) {
            operators.push(children[i].operator);
        }
        
        // 检查运算符类型并调度
        if (operators.some(op => op === '+' || op === '-')) {
            return this._processAdditionSubtractionOperation(node);
        } else if (operators.some(op => op === '*' || op === '/')) {
            return this._processMultiplicationDivisionOperation(node);
        } else if (operators.some(op => op === '^')) {
            return this._processPowerOperation(node);
        } else {
            return this._processDefaultOperation(node);
        }
    }
    
    // 括号运算处理
    _processParenthesesOperation(node) {
        const children = node.children;
        const childResult = this.rebuildExpressionFromKG(children[0]);
        return `(${childResult})`;
    }
    
    // 加减法运算处理
    _processAdditionSubtractionOperation(node) {
        const children = node.children;
        let result = this.rebuildExpressionFromKG(children[0]);
        
        for (let i = 1; i < children.length; i++) {
            const childVal = this.rebuildExpressionFromKG(children[i]);
            const operator = children[i].operator;
            result = `${result}${operator}${childVal}`;
        }
        
        return result;
    }
    
    // 乘除法运算处理
    _processMultiplicationDivisionOperation(node) {
        const children = node.children;
        let result = this.rebuildExpressionFromKG(children[0]);
        
        for (let i = 1; i < children.length; i++) {
            const childVal = this.rebuildExpressionFromKG(children[i]);
            const operator = children[i].operator;
            result = `${result}${operator}${childVal}`;
        }
        
        return result;
    }
    
    // 幂运算处理
    _processPowerOperation(node) {
        const children = node.children;
        let result = this.rebuildExpressionFromKG(children[0]);
        
        for (let i = 1; i < children.length; i++) {
            const childVal = this.rebuildExpressionFromKG(children[i]);
            const operator = children[i].operator;
            result = `${result}${operator}${childVal}`;
        }
        
        return result;
    }
    
    // 默认运算处理
    _processDefaultOperation(node) {
        const children = node.children;
        let result = this.rebuildExpressionFromKG(children[0]);
        
        for (let i = 1; i < children.length; i++) {
            const childVal = this.rebuildExpressionFromKG(children[i]);
            const operator = children[i].operator;
            
            if (operator === "()") {
                result = `(${childVal})`;
            } else {
                result = `${result}${operator}${childVal}`;
            }
        }
        
        return result;
    }
    
    // 异步处理方法
    async getVariableMethod(node) {
        await new Promise(resolve => setTimeout(resolve, 10));
        return node.expression;
    }
    
    async batchProcessSameLevel(nodes, processType) {
        const results = [];
        
        for (const node of nodes) {
            await new Promise(resolve => setTimeout(resolve, 20));
            let result;
            
            switch (processType) {
                case "llm_analysis":
                    result = `LLM分析(${node.expression})`;
                    break;
                case "web_search":
                    result = `搜索结果(${node.expression})`;
                    break;
                case "financial_analysis":
                    result = `财务数据(${node.expression})`;
                    break;
                default:
                    result = `处理结果(${node.expression})`;
            }
            
            results.push(result);
        }
        
        return results;
    }
    
    async aggregateFromLeaves(kgData) {
        const resultCache = {};
        let batchCounter = 0;
        const logLines = [];
        const executionFlow = [];
        
        while (true) {
            const readyNodes = this.findReadyForAggregation(kgData, resultCache);
            
            if (readyNodes.length === 0) {
                break;
            }
            
            batchCounter++;
            
            const currentBatchNodes = readyNodes.map(node => 
                `[${node.KG_ID.join(',')}]:${node.expression}`
            );
            
            const batchInfo = {
                batch_id: batchCounter,
                nodes: [],
                operations: []
            };
            
            for (const node of readyNodes) {
                const nodeInfo = {
                    id: node.KG_ID.join(','),
                    expression: node.expression,
                    is_variable: node.is_variable,
                    operator: node.operator || '',
                    operation_type: this.getOperationType(node)
                };
                batchInfo.nodes.push(nodeInfo);
            }
            
            executionFlow.push(batchInfo);
            
            await this.batchProcessSameLevel(readyNodes, "llm_analysis");
            
            const parentNodes = new Set();
            for (const node of readyNodes) {
                let result;
                if (node.is_variable) {
                    result = await this.getVariableMethod(node);
                } else if (node.children) {
                    result = await this.processMultiNode(node, resultCache);
                }
                
                resultCache[node.KG_ID.join(',')] = result;
                
                if (node.KG_ID.length > 1) {
                    const parentId = node.KG_ID.slice(0, -1);
                    parentNodes.add(parentId.join(','));
                }
            }
            
            const parentIdsStr = Array.from(parentNodes).sort().join(', ');
            logLines.push(`批次${batchCounter}:`);
            logLines.push("[");
            for (const nodeStr of currentBatchNodes) {
                logLines.push(`  ${nodeStr}`);
            }
            logLines.push("]");
            logLines.push("聚合后的父节点ID:");
            logLines.push(`[${parentIdsStr}]`);
            logLines.push("");
        }
        
        // 输出日志
        const logContent = [
            '# 并发任务执行批次日志\n',
            '# 格式：批次任务顺序编号, [当前批次并发处理的任务节点], [聚合后的父节点ID]\n\n',
            ...logLines.map(line => line + '\n')
        ].join('');
        
        fs.writeFileSync('.cache/output.log', logContent, 'utf8');
        
        if (this.debugMode) {
            console.log("并发任务执行日志已输出到: .cache/output.log");
        }
        
        const rootId = kgData.root.KG_ID.join(',');
        return resultCache[rootId] || "incomplete";
    }
    
    findReadyForAggregation(kgData, resultCache) {
        const readyNodes = [];
        
        const traverseNodes = (nodes, parentPath = []) => {
            for (const node of nodes) {
                const nodeId = node.KG_ID.join(',');
                
                if (nodeId in resultCache) {
                    continue;
                }
                
                if (node.is_variable) {
                    readyNodes.push(node);
                    continue;
                }
                
                if (node.children && this.childrenCompleted(node, resultCache)) {
                    readyNodes.push(node);
                    continue;
                }
                
                if (node.children) {
                    traverseNodes(node.children, node.KG_ID);
                }
            }
        };
        
        if (kgData.root) {
            if (kgData.root.children) {
                traverseNodes(kgData.root.children);
            }
            
            const rootId = kgData.root.KG_ID.join(',');
            if (!(rootId in resultCache) && this.childrenCompleted(kgData.root, resultCache)) {
                readyNodes.push(kgData.root);
            }
        }
        
        return readyNodes;
    }
    
    childrenCompleted(node, resultCache) {
        if (!node.children) {
            return true;
        }
        
        for (const child of node.children) {
            const childId = child.KG_ID.join(',');
            if (!(childId in resultCache)) {
                return false;
            }
        }
        return true;
    }
    
    getOperationType(node) {
        if (node.is_variable) {
            return "变量获取";
        }
        
        const operator = node.operator || "";
        if (operator === "()") {
            return "括号运算(概念解析)";
        } else if (['+', '-'].includes(operator)) {
            return "加减法(企业事实)";
        } else if (['*', '/'].includes(operator)) {
            return "乘除法(市场竞争)";
        } else if (operator === '^') {
            return "幂运算(行业因素)";
        } else {
            return "默认运算";
        }
    }
    
    async processMultiNode(node, cache) {
        const children = node.children;
        
        if (children.length === 1) {
            const childResult = cache[children[0].KG_ID.join(',')];
            if (children[0].operator === "()") {
                return await this.processParenthesesOperation(node, cache);
            }
            return childResult;
        }
        
        const operators = children.slice(1).map(child => child.operator);
        
        if (operators.some(op => ['+', '-'].includes(op))) {
            return await this.processAdditionSubtractionOperation(node, cache);
        } else if (operators.some(op => ['*', '/'].includes(op))) {
            return await this.processMultiplicationDivisionOperation(node, cache);
        } else if (operators.some(op => op === '^')) {
            return await this.processPowerOperation(node, cache);
        } else {
            return await this.processDefaultOperation(node, cache);
        }
    }
    
    async processParenthesesOperation(node, cache) {
        console.log("[DEBUG] 执行括号运算方法: processParenthesesOperation");
        const children = node.children;
        const childResult = cache[children[0].KG_ID.join(',')];
        return `(${childResult})`;
    }
    
    async processAdditionSubtractionOperation(node, cache) {
        console.log("[DEBUG] 执行加减法运算方法: processAdditionSubtractionOperation");
        const children = node.children;
        let result = cache[children[0].KG_ID.join(',')];
        
        for (let i = 1; i < children.length; i++) {
            const childVal = cache[children[i].KG_ID.join(',')];
            const operator = children[i].operator;
            result = `${result}${operator}${childVal}`;
        }
        
        return result;
    }
    
    async processMultiplicationDivisionOperation(node, cache) {
        console.log("[DEBUG] 执行乘除法运算方法: processMultiplicationDivisionOperation");
        const children = node.children;
        let result = cache[children[0].KG_ID.join(',')];
        
        for (let i = 1; i < children.length; i++) {
            const childVal = cache[children[i].KG_ID.join(',')];
            const operator = children[i].operator;
            result = `${result}${operator}${childVal}`;
        }
        
        return result;
    }
    
    async processPowerOperation(node, cache) {
        console.log("[DEBUG] 执行幂运算方法: processPowerOperation");
        const children = node.children;
        let result = cache[children[0].KG_ID.join(',')];
        
        for (let i = 1; i < children.length; i++) {
            const childVal = cache[children[i].KG_ID.join(',')];
            const operator = children[i].operator;
            result = `${result}${operator}${childVal}`;
        }
        
        return result;
    }
    
    async processDefaultOperation(node, cache) {
        console.log("[DEBUG] 执行默认运算方法: processDefaultOperation");
        const children = node.children;
        let result = cache[children[0].KG_ID.join(',')];
        
        for (let i = 1; i < children.length; i++) {
            const childVal = cache[children[i].KG_ID.join(',')];
            const operator = children[i].operator;
            
            if (operator === "()") {
                result = `(${childVal})`;
            } else {
                result = `${result}${operator}${childVal}`;
            }
        }
        
        return result;
    }
}

// 主函数
async function main() {
    const expression = "(A + B) * ((C - D) / (E + F) ^ G)";
    
    console.log(`解析表达式: ${expression}`);
    
    const parser = new KnowledgeGraphParser(true);
    const kgData = parser.parseToKG(expression);
    
    console.log("\n知识图谱结构:");
    console.log(JSON.stringify(kgData, null, 2));
    
    console.log("\n开始聚合处理...");
    const result = await parser.aggregateFromLeaves(kgData);
    
    console.log(`\n最终结果: ${result}`);
}

// 导出类和函数
export {
    TokenType,
    Token,
    Lexer,
    NumberNode,
    VariableNode,
    MultiOpNode,
    ParenthesesNode,
    Parser,
    KnowledgeGraphParser,
    main
};

// 如果直接运行此文件
if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}