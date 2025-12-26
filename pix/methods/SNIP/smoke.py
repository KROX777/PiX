from parsers import get_parser
params = get_parser().parse_args([])   # 用默认参数
from symbolicregression.envs.environment import FunctionEnvironment
env = FunctionEnvironment(params)

# 总数和 id 范围
print("n_words =", env.n_words)          # id 的范围是 0 .. n_words-1

# 查看某些映射示例
print("id->token sample:", list(env.equation_id2word.items())[:200])
print("token->id example: add ->", env.equation_word2id.get("add"))
print("float-sign token id: '+' ->", env.float_word2id.get("+"))
# 把模型输出的 id 列表转换为中缀表达式
# generated_seq = [...]  # 例如模型返回的 list/1D tensor
# print(env.idx_to_infix(generated_seq, is_float=False))