/**
 * Langtrain Auth Service
 * Ported from Swift LoginView.swift and AuthManager
 */

import { API_CONFIG, UserProfile } from './api';
import { secureStorage } from '../lib/storage';

// Auth State
export interface AuthState {
    isAuthenticated: boolean;
    user: UserProfile | null;
    isLoading: boolean;
    error: string | null;
}

// Login Step
export type LoginStep = 'email' | 'authenticator';

// Login Init Response
interface LoginInitResponse {
    mode: 'setup' | 'verify';
    qrCode?: string;  // Data URL for QR image
    secret?: string;  // Base32 TOTP secret
}

// TOTP Login Response
interface TOTPLoginResponse {
    redirectUrl: string;
    accessToken?: string;
    refreshToken?: string;
    userId?: string;
    email?: string;
}

// Auth Error types
export class AuthError extends Error {
    constructor(
        public code: string,
        message: string
    ) {
        super(message);
        this.name = 'AuthError';
    }

    static invalidCredentials() {
        return new AuthError('INVALID_CREDENTIALS', 'Invalid email or password');
    }

    static invalidCode() {
        return new AuthError('INVALID_CODE', 'Invalid authenticator code. Please try again.');
    }

    static userNotFound() {
        return new AuthError('USER_NOT_FOUND', 'No account found with this email. Please sign up at langtrain.xyz');
    }

    static networkError() {
        return new AuthError('NETWORK_ERROR', 'Network error. Please check your connection.');
    }

    static serverError(message: string) {
        return new AuthError('SERVER_ERROR', message);
    }
}

// Auth Manager Class (Singleton)
class AuthManager {
    private static instance: AuthManager;
    private subscribers: Set<(state: AuthState) => void> = new Set();

    private state: AuthState = {
        isAuthenticated: false,
        user: null,
        isLoading: false,
        error: null,
    };

    private constructor() {
        // Check for existing token on init
        const token = this.getToken();
        if (token) {
            this.state.isAuthenticated = true;
            // Try to fetch user profile
            this.fetchUserProfile().catch(() => {
                // Token might be invalid, clear it
                this.logout();
            });
        }
    }

    static getInstance(): AuthManager {
        if (!AuthManager.instance) {
            AuthManager.instance = new AuthManager();
        }
        return AuthManager.instance;
    }

    // State management
    getState(): AuthState {
        return { ...this.state };
    }

    subscribe(callback: (state: AuthState) => void): () => void {
        this.subscribers.add(callback);
        return () => this.subscribers.delete(callback);
    }

    private notify() {
        this.subscribers.forEach(callback => callback(this.getState()));
    }

    private setState(partial: Partial<AuthState>) {
        this.state = { ...this.state, ...partial };
        this.notify();
    }

    // Token management
    getToken(): string | null {
        return secureStorage.getItem('langtrain_auth_token');
    }

    getRefreshToken(): string | null {
        return secureStorage.getItem('langtrain_refresh_token');
    }

    private setTokens(accessToken: string, refreshToken?: string) {
        secureStorage.setItem('langtrain_auth_token', accessToken);
        if (refreshToken) {
            secureStorage.setItem('langtrain_refresh_token', refreshToken);
        }
    }

    private clearTokens() {
        secureStorage.removeItem('langtrain_auth_token');
        secureStorage.removeItem('langtrain_refresh_token');
    }

    // Auth methods
    async checkUser(email: string): Promise<{ isNewUser: boolean; qrCode?: string; secret?: string }> {
        this.setState({ isLoading: true, error: null });

        try {
            const response = await fetch(`${API_CONFIG.authURL}/login-init`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'User-Agent': 'Langtrain-Studio-Desktop/1.0',
                },
                body: JSON.stringify({ email }),
            });

            if (response.status === 404) {
                throw AuthError.userNotFound();
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw AuthError.serverError(errorData.error || 'Unknown error');
            }

            const data: LoginInitResponse = await response.json();

            this.setState({ isLoading: false });

            return {
                isNewUser: data.mode === 'setup',
                qrCode: data.qrCode,
                secret: data.secret,
            };
        } catch (error) {
            const errorMessage = error instanceof AuthError ? error.message : 'An unexpected error occurred';
            this.setState({ isLoading: false, error: errorMessage });
            throw error;
        }
    }

    async signIn(email: string, code: string): Promise<void> {
        this.setState({ isLoading: true, error: null });

        try {
            const response = await fetch(`${API_CONFIG.authURL}/totp-login-v2`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'User-Agent': 'Langtrain-Studio-Desktop/1.0',
                },
                body: JSON.stringify({
                    email,
                    code,
                    is_desktop: true,
                }),
            });

            if (response.status === 401) {
                throw AuthError.invalidCode();
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw AuthError.serverError(errorData.error || 'Unknown error');
            }

            const data: TOTPLoginResponse = await response.json();

            // Handle tokens - either from response directly or from redirect URL
            if (data.accessToken) {
                this.setTokens(data.accessToken, data.refreshToken);
                this.setState({
                    isAuthenticated: true,
                    isLoading: false,
                    user: {
                        id: data.userId || '',
                        email: data.email || email,
                        name: (data.email || email).split('@')[0],
                    },
                });
                // Fetch full profile in background
                this.fetchUserProfile().catch(console.error);
            } else if (data.redirectUrl) {
                // Handle OAuth-style redirect URL
                // Parse tokens from URL hash/query
                const url = new URL(data.redirectUrl);
                const hashParams = new URLSearchParams(url.hash.substring(1));
                const accessToken = hashParams.get('access_token') || url.searchParams.get('access_token');
                const refreshToken = hashParams.get('refresh_token') || url.searchParams.get('refresh_token');

                if (accessToken) {
                    this.setTokens(accessToken, refreshToken || undefined);
                    this.setState({
                        isAuthenticated: true,
                        isLoading: false,
                        user: {
                            id: '',
                            email,
                            name: email.split('@')[0],
                        },
                    });
                    this.fetchUserProfile().catch(console.error);
                } else {
                    // If we can't parse tokens, open the URL in browser for OAuth flow
                    // In Tauri, we'd use the shell plugin to open this
                    window.open(data.redirectUrl, '_blank');
                    this.setState({ isLoading: false });
                }
            }
        } catch (error) {
            const errorMessage = error instanceof AuthError ? error.message : 'An unexpected error occurred';
            this.setState({ isLoading: false, error: errorMessage });
            throw error;
        }
    }

    async fetchUserProfile(): Promise<void> {
        try {
            const response = await fetch(`${API_CONFIG.apiURL}/auth/me`, {
                headers: {
                    'Authorization': `Bearer ${this.getToken()}`,
                    'User-Agent': 'Langtrain-Studio-Desktop/1.0',
                },
            });

            if (response.ok) {
                const user: UserProfile = await response.json();
                this.setState({ user });
            }
        } catch (error) {
            console.error('Failed to fetch user profile:', error);
        }
    }

    logout(): void {
        this.clearTokens();
        secureStorage.clear();
        sessionStorage.clear();
        this.setState({
            isAuthenticated: false,
            user: null,
            error: null,
        });
    }
}

export const authManager = AuthManager.getInstance();

// React hook for auth state
export function useAuth(): AuthState & {
    checkUser: (email: string) => Promise<{ isNewUser: boolean; qrCode?: string; secret?: string }>;
    signIn: (email: string, code: string) => Promise<void>;
    logout: () => void;
} {
    const [state, setState] = React.useState(authManager.getState());

    React.useEffect(() => {
        return authManager.subscribe(setState);
    }, []);

    return {
        ...state,
        checkUser: (email) => authManager.checkUser(email),
        signIn: (email, code) => authManager.signIn(email, code),
        logout: () => authManager.logout(),
    };
}

// Need to import React for the hook
import * as React from 'react';
