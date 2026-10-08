// Chinese for the DevKit error codes (scripts/atlas/errors.ts CODES), shown on /devkit/contribute/ next to the English
// meaning and the explanation from site/DEVKIT.md section 13. `en` is the English meaning each translation was written
// for: when a code is added or its meaning changes in errors.ts, the build fails (check.ts) until this table follows.
// Groups are the hundreds digit (AK2xx: parity).

export const CODE_ZH: Record<string, { en: string; zh: string }> = {
	AK001: { en: 'Node or npm', zh: 'Node 或 npm' },
	AK002: { en: 'Python or Python packages', zh: 'Python 或 Python 包' },
	AK003: { en: 'port busy', zh: '端口被占用' },
	AK004: { en: 'git refuses the repo', zh: 'git 拒绝这个仓库' },
	AK101: { en: 'graph differs between cases', zh: '不同用例记录的计算图不同' },
	AK102: { en: 'CUDA or TF32 during export', zh: '导出时用了 CUDA 或 TF32' },
	AK103: { en: 'describe() differs from module(x)', zh: 'describe() 与 module(x) 不一致' },
	AK104: { en: 'recurrent gate reference differs from the module', zh: '循环网络门控的参考值与模块不一致' },
	AK105: { en: 'budget at export', zh: '导出时超出体积预算' },
	AK106: { en: 'op attributes differ from ops.json', zh: '算子属性与 ops.json 不符' },
	AK201: { en: 'layer over the parity bar (kind A)', zh: '某一层超出一致性阈值（A 类）' },
	AK202: { en: 'explain() differs from the trace', zh: 'explain() 与轨迹不一致' },
	AK203: { en: 'twin mismatch (kind B)', zh: '对照训练不一致（B 类）' },
	AK204: { en: 'reference mismatch (kind C, first divergence)', zh: '参考实现不一致（C 类，报告第一个分歧）' },
	AK205: { en: 'near-tie divergence (warning)', zh: '近乎平局导致的分歧（警告）' },
	AK301: { en: 'schema', zh: '格式（schema）问题' },
	AK302: { en: 'inferShape differs from outShape at a dim extreme', zh: '维度取极值时 inferShape 与 outShape 不一致' },
	AK303: { en: 'size budget', zh: '体积预算' },
	AK401: { en: 'placeholder unresolved or non-finite', zh: '占位符没有值，或不是有限数' },
	AK402: { en: 'literal multi-digit number in narration (warning)', zh: '旁白里手写了多位数字（警告）' },
	AK403: { en: 'en/zh placeholder sets differ', zh: '中英文旁白的占位符不同' },
	AK404: { en: 'cursor not found on a fixture', zh: '某个测试输入上找不到这一步的帧' },
	AK405: { en: 'view not registered', zh: '视图没有注册' },
	AK406: { en: 'missing zh', zh: '缺少中文' },
	AK407: { en: 'lesson shape', zh: '课程结构' },
	AK408: { en: 'claim test failed', zh: '论断测试失败' },
	AK409: { en: 'reviewed lesson made only of auto steps', zh: '已评审的课程只有自动步骤' },
	AK410: { en: 'two different numbers print the same (warning)', zh: '两个不同的数打印出来一样（警告）' },
	AK411: { en: 'step term in no glossary', zh: '步骤用的术语不在任何术语表里' },
	AK412: { en: 'predict narration gives the answer away (warning)', zh: '预测题的旁白泄露了答案（警告）' },
	AK501: { en: 'TODO markers in the manifest', zh: '清单里还有 TODO 标记' },
	AK502: { en: 'step without provenance', zh: '步骤没有出处徽章' },
	AK503: { en: 'dependency rule or Math.random', zh: '违反依赖规则，或用了 Math.random' },
	AK504: { en: 'empty describe()', zh: 'describe() 为空' },
	AK601: { en: 'console error', zh: '浏览器控制台报错' },
	AK602: { en: 'horizontal scroll at 360 px', zh: '360 像素宽时出现横向滚动' },
	AK603: { en: 'long task', zh: '主线程长任务' },
	AK604: { en: 'accessibility violation (axe: serious or critical)', zh: '无障碍问题（axe：严重或关键级别）' },
	AK701: { en: 'slug not in catalog', zh: 'slug 不在目录中' },
	AK702: { en: 'change outside owned paths', zh: '改动超出车道拥有的路径' },
	AK703: { en: 'sync refused', zh: '同步被拒绝' },
	AK704: { en: 'credits missing', zh: '缺少致谢信息' }
};

export const CODE_GROUPS: Record<string, { en: string; zh: string }> = {
	'0': { en: 'Environment', zh: '环境' },
	'1': { en: 'Export (kind A)', zh: '导出（A 类）' },
	'2': { en: 'Parity', zh: '一致性' },
	'3': { en: 'Schema and budgets', zh: '格式与预算' },
	'4': { en: 'Lesson', zh: '课程' },
	'5': { en: 'Honesty and the manifest', zh: '诚实性与清单' },
	'6': { en: 'Browser test', zh: '浏览器测试' },
	'7': { en: 'Catalog and lanes', zh: '目录与车道' }
};

export const groupOf = (code: string) => code.slice(2, 3);
