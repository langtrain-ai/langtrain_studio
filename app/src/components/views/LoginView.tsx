/**
 * Login View Component
 * Ported from Swift LoginView.swift
 */

import { useState } from 'react';
import { Cpu, Lock, RefreshCw } from 'lucide-react';
import { useAuth, type LoginStep } from '../../services/auth';
import './LoginView.css';

interface FeatureRowProps {
    icon: React.ReactNode;
    text: string;
}

function FeatureRow({ icon, text }: FeatureRowProps) {
    return (
        <div className="login__feature-row">
            <span className="login__feature-icon">{icon}</span>
            <span className="login__feature-text">{text}</span>
        </div>
    );
}

export function LoginView() {
    const { isLoading, error, checkUser, signIn } = useAuth();

    const [step, setStep] = useState<LoginStep>('email');
    const [email, setEmail] = useState('');
    const [code, setCode] = useState('');
    const [isNewUser, setIsNewUser] = useState(false);
    const [qrCode, setQrCode] = useState<string | null>(null);
    const [secret, setSecret] = useState('');

    const isEmailValid = email.includes('@') && email.includes('.');
    const isCodeValid = code.length >= 6;

    const handleCheckUser = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!isEmailValid) return;

        try {
            const result = await checkUser(email);
            setIsNewUser(result.isNewUser);
            setQrCode(result.qrCode || null);
            setSecret(result.secret || '');
            setStep('authenticator');
        } catch (err) {
            // Error is handled by auth service
        }
    };

    const handleSignIn = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!isCodeValid) return;

        try {
            await signIn(email, code);
            // Navigation will be handled by App.tsx based on auth state
        } catch (err) {
            // Error is handled by auth service
        }
    };

    const handleBack = () => {
        setStep('email');
        setCode('');
        setIsNewUser(false);
        setQrCode(null);
        setSecret('');
    };

    const copySecret = () => {
        navigator.clipboard.writeText(secret);
    };

    return (
        <div className="login">
            {/* Left Panel - Branding */}
            <div className="login__branding">
                <div className="login__branding-content">
                    <div className="login__logo">
                        <img
                            src="/langtrain-logo.svg"
                            alt="Langtrain"
                            className="login__logo-icon"
                        />
                    </div>

                    <h1 className="login__title">Langtrain Studio</h1>
                    <p className="login__tagline">Custom LLMs: Finetune, Train, Deploy</p>

                    <div className="login__features">
                        <FeatureRow icon={<Cpu size={14} />} text="Local & Cloud Training" />
                        <FeatureRow icon={<Lock size={14} />} text="Secure Authentication" />
                        <FeatureRow icon={<RefreshCw size={14} />} text="Seamless Model Sync" />
                    </div>
                </div>
            </div>

            {/* Right Panel - Form */}
            <div className="login__form-panel">
                <div className="login__form-container">
                    <h2 className="login__form-title">
                        {step === 'email' ? 'Sign In' : 'Enter Code'}
                    </h2>
                    <p className="login__form-subtitle">
                        {step === 'email'
                            ? 'Access your Langtrain account'
                            : 'Enter your authenticator code'}
                    </p>

                    {step === 'email' ? (
                        <form onSubmit={handleCheckUser} className="login__form">
                            <div className="login__field">
                                <label className="login__label">Email</label>
                                <input
                                    type="email"
                                    className="login__input"
                                    placeholder="name@company.com"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    autoFocus
                                />
                            </div>

                            <button
                                type="submit"
                                className="login__button login__button--primary"
                                disabled={!isEmailValid || isLoading}
                            >
                                {isLoading ? 'Loading...' : 'Continue'}
                            </button>
                        </form>
                    ) : (
                        <form onSubmit={handleSignIn} className="login__form">
                            {isNewUser && qrCode && (
                                <div className="login__qr-section">
                                    <p className="login__qr-title">Set up Authenticator</p>
                                    <img
                                        src={qrCode}
                                        alt="QR Code"
                                        className="login__qr-code"
                                    />
                                    <p className="login__qr-hint">Scan with your authenticator app</p>

                                    {secret && (
                                        <div className="login__secret">
                                            <span className="login__secret-label">Or enter manually:</span>
                                            <div className="login__secret-value">
                                                <code>{secret}</code>
                                                <button
                                                    type="button"
                                                    className="login__secret-copy"
                                                    onClick={copySecret}
                                                >
                                                    Copy
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            <div className="login__field">
                                <label className="login__label">Authenticator Code</label>
                                <input
                                    type="text"
                                    className="login__input login__input--code"
                                    placeholder="123456"
                                    value={code}
                                    onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                                    autoFocus
                                    maxLength={6}
                                />
                            </div>

                            <button
                                type="submit"
                                className="login__button login__button--primary"
                                disabled={!isCodeValid || isLoading}
                            >
                                {isLoading
                                    ? 'Loading...'
                                    : isNewUser
                                        ? 'Verify & Create Account'
                                        : 'Sign In'}
                            </button>

                            <button
                                type="button"
                                className="login__button login__button--text"
                                onClick={handleBack}
                            >
                                ← Back to Email
                            </button>
                        </form>
                    )}

                    {error && (
                        <div className="login__error">
                            {error}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
