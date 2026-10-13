// The UI icons of the DevKit pages, copied from $lib/kit/icons (same 24 grid, drawn by $lib/kit/Icon.svelte).
// They are copied, not imported: kit/icons.ts sits in the site's shared shell chunk (the header uses it), and every
// export a DevKit page imported from it would grow the shell of every page, the planned model pages (which have about
// 0.4 KB of budget left) included. Here they load with the DevKit pages only. Keep them in step with kit/icons.ts.
export const x = '<path d="M6 6l12 12M18 6L6 18"/>';
export const left = '<path d="M15 5l-7 7 7 7"/>';
export const right = '<path d="M9 5l7 7-7 7"/>';
export const arrow = '<path d="M5 12h14M13 6l6 6-6 6"/>';
export const list =
	'<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1" fill="currentColor"/><circle cx="4.5" cy="12" r="1" fill="currentColor"/><circle cx="4.5" cy="18" r="1" fill="currentColor"/>';
export const code = '<path d="M8.5 7L3.5 12l5 5M15.5 7l5 5-5 5"/>';
export const lesson = '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M10 9l5 3-5 3z" fill="currentColor"/>';
export const info = '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5h.01"/>';
export const flask = '<path d="M9 3h6M10 3v6l-5 9.5A1.5 1.5 0 0 0 6.3 21h11.4a1.5 1.5 0 0 0 1.3-2.5L14 9V3"/>';
export const quiz = '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M8 12.5l2.5 2.5L16 9.5"/>';
