# Lint 错误处理与增量治理指南

当项目中已经存在大量的 ESLint 报错或警告时，`npm run lint` 往往会直接失败，影响当前改动的提交流程。本指南提供一套可逐步落地的治理方案，帮助团队在不影响日常开发的情况下逐步修复历史问题。

## 1. 只检查本次变更的文件

### 使用 lint-staged

1. 安装依赖：
   ```bash
   npm install --save-dev lint-staged husky
   npx husky install
   ```
2. 在 `package.json` 中新增 lint-staged 配置，只对暂存区文件执行检查：
   ```json
   {
     "lint-staged": {
       "*.{js,jsx,ts,tsx}": [
         "eslint --cache --max-warnings=0",
         "eslint --cache --fix",
         "git add"
       ]
     }
   }
   ```
3. 添加预提交钩子（`.husky/pre-commit`）：
   ```bash
   #!/bin/sh
   . "$(dirname "$0")/_/husky.sh"

   npx lint-staged
   ```

通过以上配置，`git commit` 时只会 lint 当前提交所修改的文件，实现对新问题的“零容忍”。

### 直接在命令行限制范围

如果暂时不想引入 lint-staged，可以手动执行：
```bash
# 仅 lint 最近一次提交中的文件
git diff --name-only HEAD~1 | xargs npx eslint --cache --max-warnings=0
```

## 2. 加速与稳定 lint 体验

- `eslint --cache`：缓存 lint 结果，只重新检查被修改的文件。
- `eslint --max-warnings=0`：CI 中将警告视为失败，避免新警告混入代码库。
- `eslint --fix` / `eslint --fix-type problem`：自动修复可安全处理的问题类型。
- 对执行耗时较长的项目，可以考虑把 lint 与测试拆分到不同的 CI 任务中并行执行。

## 3. 将错误降级为可控的警告

短期内无法全部修复的规则可以采用以下策略：

- [eslint-plugin-only-warn](https://github.com/bfanger/eslint-plugin-only-warn)：在遗留代码上临时将 error 降级为 warning，CI 中可以允许警告存在。
- `.eslintignore`：把暂时无法治理的旧目录排除在 lint 之外，并在文档中记录后续治理计划。
- 目录级覆盖规则：通过 `overrides` 为遗留目录放宽规则，对新目录执行严格规则。

## 4. 为遗留问题建立基线

当历史错误非常多时，可以生成“基线”文件，保证 CI 只关注新增的问题：

1. 导出一次性报告供人工排查：
   ```bash
   npx eslint "src/**/*.{js,ts,tsx}" \
     --format json \
     --output-file eslint-baseline.json
   ```
2. 在 CI 中读取该文件过滤掉既有的错误，或使用社区方案（如 [eslint-baseline](https://github.com/cletusw/eslint-baseline)）自动忽略历史问题。
3. 每次修复后更新基线文件，直至完全清零。

## 5. CI 执行策略建议

- **增量检查（强制通过）**：只检查 PR 中新增或修改的文件，并启用 `--max-warnings=0`，确保新问题不会进入代码库。
- **全量基线检查（允许失败）**：保留一次全量 lint 任务，结合基线文件跟踪整体问题数量，可在 CI 中设置为非阻塞以便持续观察。

## 6. 逐步清理遗留问题的建议流程

1. **统计与归类**：通过 `eslint --format codeframe` 或 `eslint --format json` 了解问题类型，优先治理安全类问题（如 `problem` 类型）。
2. **自动化优先**：尝试 `eslint --fix` 与 `eslint --fix-type problem`。对于无法自动修复的场景，通过脚本或 codemod 辅助。
3. **分目录推进**：按照模块拆分治理计划，结合 `.eslintignore` 或 `overrides`，逐渐提高规则严格度。
4. **文档化约定**：在团队开发手册中说明 lint 流程，确保每位开发者都能正确运行增量检查。

## 7. 常见问题排查

| 问题 | 解决思路 |
| --- | --- |
| `eslint` 在 CI 上过慢 | 启用 `--cache`、在流水线上持久化缓存；或拆分多个任务并行执行 |
| 本地与 CI 检查结果不一致 | 确保使用统一的 Node 版本与依赖；在 `package.json` 中锁定 eslint 相关版本 |
| 某些第三方库类型缺失导致报错 | 在对应目录配置 `tsconfig.eslint.json`，或在 `overrides` 中关闭相关规则 |
| 需要对部分文件放宽规则 | 使用文件顶部的 `/* eslint-disable */`，同时添加 TODO 说明后续处理计划 |

---

通过以上策略，可以在不阻塞当前需求开发的前提下，逐步把 lint 问题控制住，实现从“能提交”到“零警告”的平滑过渡。
