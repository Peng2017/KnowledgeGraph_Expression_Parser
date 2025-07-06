import { KnowledgeGraphParser } from './KnowledgeGraph_Expression_Parser.js';
import fs from 'fs';
import path from 'path';

// 确保.cache目录存在
const cacheDir = '.cache';
if (!fs.existsSync(cacheDir)) {
    fs.mkdirSync(cacheDir, { recursive: true });
}

// 主测试函数
async function runTest() {
    // 测试表达式
    const testExpression = "A1^2+((B1*2-C1)/D1^(E+F)+G)*H-1";
    
    console.log(`=== 综合测试程序 ===`);
    console.log(`测试表达式: ${testExpression}`);
    console.log(`开始解析和重建测试...\n`);
    
    // 创建解析器实例
    const parser = new KnowledgeGraphParser(true);
    
    // 1. 解析表达式为知识图谱
    console.log("1. 解析表达式为知识图谱...");
    const kgData = parser.parseToKG(testExpression);
    
    // 2. 从知识图谱重建表达式
    console.log("\n2. 从知识图谱重建表达式...");
    const rebuiltExpression = await parser.aggregateFromLeaves(kgData);

    // 3. 对比结果
    console.log("\n3. 对比结果:");
    console.log(`原始表达式: ${testExpression}`);
    console.log(`重建表达式: ${rebuiltExpression}`);
    console.log(`表达式一致: ${testExpression === rebuiltExpression}`);
    
    // 4. 生成输出文件
    console.log("\n4. 生成输出文件...");
    
    // 生成对比日志
    const compareLog = `=== 表达式解析与重建对比测试 ===\n` +
        `测试时间: ${new Date().toLocaleString()}\n\n` +
        `原始表达式: ${testExpression}\n` +
        `重建表达式: ${rebuiltExpression}\n` +
        `表达式长度: 原始=${testExpression.length}, 重建=${rebuiltExpression.length}\n` +
        `表达式一致: ${testExpression === rebuiltExpression}\n\n` +
        `=== 知识图谱根节点信息 ===\n` +
        `${JSON.stringify(kgData.root, null, 2)}\n`;
    
    fs.writeFileSync(path.join(cacheDir, 'compare.log'), compareLog);
    
    // 生成AST结构文件
    const astOutput = `=== 抽象语法树(AST)结构 ===\n` +
        `表达式: ${testExpression}\n` +
        `解析器类型: KnowledgeGraphParser\n\n` +
        `=== 知识图谱完整结构 ===\n` +
        `${JSON.stringify(kgData, null, 2)}\n\n` +
        `=== 运算符优先级表 ===\n` +
        `加减法 (+, -): 优先级 20\n` +
        `乘除法 (*, /): 优先级 40\n` +
        `幂运算 (^): 优先级 60\n`;
    
    fs.writeFileSync(path.join(cacheDir, 'knowledge_graph.ast'), astOutput);
    
    // 生成处理日志（使用不同文件名避免覆盖）
    const processLog = `=== 表达式解析处理日志 ===\n` +
        `表达式: ${testExpression}\n` +
        `处理时间: ${new Date().toLocaleString()}\n\n` +
        `=== 解析步骤 ===\n` +
        `1. 词法分析: 将表达式分解为Token序列\n` +
        `2. 语法分析: 构建抽象语法树(AST)\n` +
        `3. 知识图谱转换: AST转换为多叉树结构\n` +
        `4. 表达式重建: 从叶子节点递归聚合\n\n` +
        `=== 重建结果 ===\n` +
        `最终聚合结果: ${rebuiltExpression}\n`;
    
    fs.writeFileSync(path.join(cacheDir, 'process.log'), processLog);
    
    console.log("\n=== 测试完成 ===");
    console.log("输出文件已生成:");
    console.log("- .cache/compare.log: 表达式对比结果");
    console.log("- .cache/knowledge_graph.ast: 知识图谱AST结构信息");
    console.log("- .cache/process.log: 表达式解析处理日志");
}

// 运行测试
runTest().catch(console.error);