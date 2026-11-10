/**
 * Secure Storage Stub for Electron
 * 
 * In a true production Electron app, this should interface via IPC (`window.electronAPI.keychain`)
 * to `safeStorage` or `keytar` to encrypt tokens securely at rest using os-native keychains.
 */

export const secureStorage = {
    getItem: (key: string): string | null => {
        // TODO: Replace with secure IPC call (e.g. window.electron.keychain.get(key))
        return localStorage.getItem(key);
    },
    setItem: (key: string, value: string): void => {
        // TODO: Replace with secure IPC call
        localStorage.setItem(key, value);
    },
    removeItem: (key: string): void => {
        // TODO: Replace with secure IPC call
        localStorage.removeItem(key);
    },
    clear: (): void => {
        // TODO: Replace with secure IPC call
        localStorage.clear();
    }
};
