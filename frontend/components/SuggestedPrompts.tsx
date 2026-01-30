'use client'

interface SuggestedPromptsProps {
    onSelectPrompt: (prompt: string) => void
}

const SUGGESTED_PROMPTS = [
    {
        emoji: '🔍',
        title: 'Tìm Python Developer',
        prompt: 'Tìm ứng viên Python Developer có 3 năm kinh nghiệm ở Hà Nội',
    },
    {
        emoji: '👨‍💻',
        title: 'Java Senior',
        prompt: 'Tìm Java Developer senior có kinh nghiệm với Spring Boot và Microservices',
    },
    {
        emoji: '📊',
        title: 'Data Engineer',
        prompt: 'Tìm Data Engineer biết Apache Spark và có kinh nghiệm xây dựng data pipeline',
    },
    {
        emoji: '🎨',
        title: 'Frontend React',
        prompt: 'Tìm Frontend Developer chuyên React.js với 2+ năm kinh nghiệm',
    },
    {
        emoji: '📱',
        title: 'Mobile Developer',
        prompt: 'Tìm Mobile Developer biết Flutter hoặc React Native',
    },
    {
        emoji: '☁️',
        title: 'DevOps Engineer',
        prompt: 'Tìm DevOps Engineer có kinh nghiệm với AWS và Kubernetes',
    },
]

export default function SuggestedPrompts({
    onSelectPrompt,
}: SuggestedPromptsProps) {
    return (
        <div className="w-full max-w-3xl">
            <p className="text-sm text-gray-400 mb-4 text-center">
                💡 Gợi ý câu hỏi để bắt đầu:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {SUGGESTED_PROMPTS.map((item, index) => (
                    <button
                        key={index}
                        onClick={() => onSelectPrompt(item.prompt)}
                        className="group p-4 bg-gray-800/50 hover:bg-gray-700/50 border border-gray-700/50 hover:border-primary-500/50 rounded-xl text-left transition-all duration-200"
                    >
                        <div className="text-2xl mb-2">{item.emoji}</div>
                        <h3 className="font-medium text-white text-sm group-hover:text-primary-400 transition-colors">
                            {item.title}
                        </h3>
                        <p className="text-xs text-gray-500 mt-1 line-clamp-2">
                            {item.prompt}
                        </p>
                    </button>
                ))}
            </div>
        </div>
    )
}
