// The {% %} template renderer used by `atlas new`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { render } from '../lib/template.ts';

test('variables, sections, inverse sections, line-swallowing block tags', () => {
	const tpl = 'a {%x%}\n{%#on%}\nkept {%x%}\n{%/on%}\n{%#off%}\ndropped\n{%/off%}\n{%^off%}inverse{%/off%} {{kitSlot}} {name}\n';
	assert.equal(render(tpl, { x: 1, on: true, off: false }), 'a 1\nkept 1\ninverse {{kitSlot}} {name}\n');
});

test('nested sections of different names', () => {
	assert.equal(render('{%#a%}\nA\n{%#b%}\nB\n{%/b%}\n{%/a%}\nend', { a: true, b: false }), 'A\nend');
	assert.equal(render('{%#a%}\nA\n{%#b%}\nB\n{%/b%}\n{%/a%}\nend', { a: true, b: true }), 'A\nB\nend');
	assert.equal(render('{%#a%}\nA\n{%#b%}\nB\n{%/b%}\n{%/a%}\nend', { a: false, b: true }), 'end');
});

test('unknown names and unbalanced tags throw', () => {
	assert.throws(() => render('{%nope%}', {}), /unknown variable/);
	assert.throws(() => render('{%#nope%}x{%/nope%}', {}), /unknown section/);
	assert.throws(() => render('{%#a%}x', { a: true }), /unbalanced/);
});
