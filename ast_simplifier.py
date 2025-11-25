import javalang
import re


class JavaCodeSimplifier:
    def __init__(self):
        # 定义不需要保留的 Token 类型或具体内容 (停用词列表)
        # Level 3: 最先被删除
        self.useless_tokens = {
            ';', '{', '}', '(', ')', '[', ']', '.', ',', '@',
            'public', 'private', 'protected', 'static', 'final',
            'volatile', 'synchronized', 'native', 'transient',
        }

    def tokenize_code(self, code):
        """
        使用 javalang 对代码进行 Tokenize
        """
        try:
            # javalang.tokenizer.tokenize 返回的是生成器，转为列表
            tokens = list(javalang.tokenizer.tokenize(code))
            return tokens
        except javalang.tokenizer.LexerError:
            # 如果代码片段不完整（常见于数据集），尝试简单的正则分词作为回退
            return self._fallback_tokenizer(code)

    def _fallback_tokenizer(self, code):
        """
        简单的正则分词，用于处理 javalang 解析失败的情况
        """
        # 简单的正则匹配单词和符号
        token_strs = re.findall(r"[\w']+|[.,!?;{}[\]()]", code)

        # 模拟 javalang 的 Token 对象结构，只需要 value 属性
        class MockToken:
            def __init__(self, value):
                self.value = value

        return [MockToken(t) for t in token_strs]

    def assign_importance(self, token):
        """
        根据 Token 类型和内容分配重要性分数。
        分数越高越重要，越不容易被删除。
        """
        value = token.value

        # --- Level 1: 最高优先级 (保留核心逻辑) ---
        # 1. 基本类型和值
        if isinstance(token, (javalang.tokenizer.Integer,
                              javalang.tokenizer.FloatingPoint,
                              javalang.tokenizer.String)):
            return 3

        if isinstance(value, str) and re.fullmatch(r"[0-9]+(\.[0-9]+)?", value):
            return 3

        # 2. 标识符 (变量名, 方法名, 类名)
        # 排除单字符变量名 (通常是 i, j, k 等循环变量，有时为了压缩可以牺牲)
        if isinstance(token, javalang.tokenizer.Identifier):
            if len(value) > 1:
                return 3
            return 2

        # 3. 核心控制流关键字
        if value in {'if', 'else', 'for', 'while', 'return', 'switch', 'case', 'break', 'continue', 'try', 'catch'}:
            return 3

        # 4. 关键运算符
        if value in {'=', '==', '!=', '+', '-', '*', '/', '<', '>', '&&', '||'}:
            return 2

        # --- Level 2: 中等优先级 ---
        # 基础类型关键字
        if value in {'int', 'float', 'double', 'boolean', 'char', 'String', 'void', 'class'}:
            return 2

        # --- Level 3: 低优先级 (可删除) ---
        # 包含修饰符、标点符号等
        if value in self.useless_tokens:
            return 1

        # 默认低优先级
        return 1

    def simplify(self, code, remove_ratio=0.3):
        """
        核心简化函数
        :param code: 原始 Java 代码
        :param remove_ratio: 删除比例 (0.0 - 1.0)
        """
        # 全局兜底，确保不抛异常到外面
        try:
            # 统一转成字符串，防止传入 None/数字等
            code_str = "" if code is None else str(code)

            if not code_str:
                return ""

            tokens = self.tokenize_code(code_str)
            total_tokens = len(tokens)

            if total_tokens == 0:
                return ""

            num_to_remove = int(total_tokens * remove_ratio)
            if num_to_remove == 0:
                return " ".join([str(t.value) for t in tokens])

            scored_tokens = []
            for i, token in enumerate(tokens):
                # 再防一层，确保有 value
                value = getattr(token, "value", "")
                score = self.assign_importance(token)
                scored_tokens.append({
                    "index": i,
                    "token": token,
                    "value": value,
                    "score": score,
                })

            scored_tokens.sort(key=lambda x: x["score"])

            indices_to_remove = {scored_tokens[i]["index"] for i in range(num_to_remove)}

            simplified_tokens = []
            for i, token in enumerate(tokens):
                if i not in indices_to_remove:
                    simplified_tokens.append(str(getattr(token, "value", "")))

            return " ".join(simplified_tokens)
        except Exception:
            # 任意异常都退回原始字符串
            return "" if code is None else str(code)