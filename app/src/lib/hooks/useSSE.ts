import { useEffect, useState } from 'react';

export interface SSEEvent {
    type: string;
    project_id: string;
    timestamp: string;
    payload?: any;
    resource_id?: string;
    status?: string;
    [key: string]: any;
}

/**
 * Connects Desktop App to Langtrain Data Plane SSE stream.
 * Automatically handles reconnection and JSON parsing.
 */
export function useSSE(projectId: string | undefined, eventType?: string) {
    const [lastEvent, setLastEvent] = useState<SSEEvent | null>(null);
    const [error, setError] = useState<Error | null>(null);
    const [isConnected, setIsConnected] = useState(false);

    useEffect(() => {
        if (!projectId) return;

        // Direct connection to Langtrain Data Plane
        // Vite uses import.meta.env
        const baseUrl = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';
        const url = `${baseUrl}/api/v1/events?project_id=${projectId}`;

        // Desktop auth often uses JWT headers, but EventSource doesn't support custom headers natively.
        // Assuming backend accepts `withCredentials` or auth is passed via query params 
        // for Electron native requests utilizing cookies.
        const eventSource = new EventSource(url, {
            withCredentials: true
        });

        eventSource.onopen = () => {
            setIsConnected(true);
            setError(null);
        };

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (!eventType || data.type === eventType) {
                    setLastEvent(data);
                }
            } catch (err: any) {
                console.error("Failed to parse SSE event:", err);
            }
        };

        eventSource.onerror = (err: any) => {
            console.error("SSE connection error:", err);
            setError(err);
            setIsConnected(false);
        };

        return () => {
            eventSource.close();
            setIsConnected(false);
        };
    }, [projectId, eventType]);

    return { lastEvent, isConnected, error };
}
