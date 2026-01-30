'use client'

import { useRef, useEffect } from 'react'
import { Message } from '@/lib/types'
import MessageBubble from './MessageBubble'
import ThinkingIndicator from './ThinkingIndicator'

interface ChatWindowProps {
    messages: Message[]
    isThinking: boolean
    thinkingStatus?: string
}

export default function ChatWindow({ messages, isThinking, thinkingStatus }: ChatWindowProps) {
    const bottomRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, isThinking])

    return (
        <div className="h-full overflow-y-auto px-4 py-6">
            <div className="max-w-4xl mx-auto space-y-4">
                {messages.map((message) => (
                    <MessageBubble key={message.id} message={message} />
                ))}

                {isThinking && <ThinkingIndicator message={thinkingStatus} />}

                <div ref={bottomRef} />
            </div>
        </div>
    )
}
