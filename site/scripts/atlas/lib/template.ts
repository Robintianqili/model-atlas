// A tiny template renderer for scripts/atlas/templates/**. The delimiters are {% %} because the generated code itself
// uses {{name}} (kit TeX slots) and {name} (narration placeholders).
//   {%name%}              the value (an unknown name throws: templates never ship a silent hole)
//   {%#flag%}...{%/flag%} kept when flag is truthy;  {%^flag%}...{%/flag%} kept when it is falsy
// A section tag alone on its line takes its line break with it, so sections do not leave blank lines.
export type Vars = Record<string, string | number | boolean | undefined>;

export function render(tpl: string, vars: Vars, file = 'template'): string {
	let out = tpl.replace(/\r\n/g, '\n');
	const section = /^[ \t]*\{%([#^])(\w+)%\}[ \t]*\n([\s\S]*?)^[ \t]*\{%\/\2%\}[ \t]*\n|\{%([#^])(\w+)%\}([\s\S]*?)\{%\/\5%\}/m;
	for (let guard = 0; guard < 1000; guard++) {
		const m = section.exec(out);
		if (!m) break;
		const [kind, name, body] = m[1] ? [m[1], m[2], m[3]] : [m[4], m[5], m[6]];
		if (!(name in vars)) throw new Error(`${file}: unknown section {%${kind}${name}%}`);
		const on = Boolean(vars[name]);
		out = out.slice(0, m.index) + ((kind === '#') === on ? body : '') + out.slice(m.index + m[0].length);
	}
	out = out.replace(/\{%(\w+)%\}/g, (_, name: string) => {
		if (!(name in vars) || vars[name] === undefined) throw new Error(`${file}: unknown variable {%${name}%}`);
		return String(vars[name]);
	});
	const left = /\{%[^%]*%\}/.exec(out);
	if (left) throw new Error(`${file}: unbalanced template tag ${left[0]}`);
	return out;
}
