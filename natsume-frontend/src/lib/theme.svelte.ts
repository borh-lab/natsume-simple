import { browser } from '$app/environment';

export const themeManager = (() => {
	let darkMode = $state(false);

	function setDarkMode(value: boolean) {
		darkMode = value;
		if (browser) {
			if (darkMode) {
				document.documentElement.classList.add('dark');
			} else {
				document.documentElement.classList.remove('dark');
			}
		}
	}

	return {
		get isDarkMode() {
			return darkMode;
		},
		toggle() {
			setDarkMode(!darkMode);
		},
		set isDarkMode(value: boolean) {
			setDarkMode(value);
		}
	};
})();
