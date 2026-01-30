'use client'

import { CandidateCard as CandidateCardType } from '@/lib/types'

interface CandidateCardProps {
    candidate: CandidateCardType
}

export default function CandidateCard({ candidate }: CandidateCardProps) {
    const handleClick = () => {
        // Open candidate detail in new tab or modal
        window.open(`/candidates/${candidate.candidate_id}`, '_blank')
    }

    return (
        <div
            onClick={handleClick}
            className="bg-gray-900/60 hover:bg-gray-900/80 border border-gray-600/50 hover:border-primary-500/50 rounded-xl p-3 cursor-pointer transition-all duration-200 group"
        >
            <div className="flex items-start justify-between gap-3">
                {/* Avatar and Info */}
                <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-400 to-accent-500 flex items-center justify-center text-white font-bold text-sm shrink-0">
                        {candidate.full_name
                            .split(' ')
                            .map((n) => n[0])
                            .slice(-2)
                            .join('')
                            .toUpperCase()}
                    </div>
                    <div className="min-w-0">
                        <h4 className="font-medium text-white text-sm group-hover:text-primary-400 transition-colors truncate">
                            {candidate.full_name}
                        </h4>
                        {candidate.headline && (
                            <p className="text-xs text-gray-400 truncate">
                                {candidate.headline}
                            </p>
                        )}
                        {candidate.total_experience_years && (
                            <p className="text-xs text-gray-500 mt-1">
                                ⏱️ {candidate.total_experience_years.toFixed(1)} năm kinh nghiệm
                            </p>
                        )}
                    </div>
                </div>

                {/* Match Score */}
                {candidate.match_score && (
                    <div className="shrink-0">
                        <div
                            className={`px-2 py-1 rounded-lg text-xs font-medium ${candidate.match_score >= 0.7
                                    ? 'bg-green-500/20 text-green-400'
                                    : candidate.match_score >= 0.4
                                        ? 'bg-yellow-500/20 text-yellow-400'
                                        : 'bg-gray-500/20 text-gray-400'
                                }`}
                        >
                            {(candidate.match_score * 100).toFixed(0)}%
                        </div>
                    </div>
                )}
            </div>

            {/* Skills */}
            {candidate.top_skills && candidate.top_skills.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-2">
                    {candidate.top_skills.slice(0, 4).map((skill, index) => (
                        <span
                            key={index}
                            className="px-2 py-0.5 bg-gray-700/50 text-gray-300 text-xs rounded-md"
                        >
                            {skill}
                        </span>
                    ))}
                    {candidate.top_skills.length > 4 && (
                        <span className="px-2 py-0.5 text-gray-500 text-xs">
                            +{candidate.top_skills.length - 4}
                        </span>
                    )}
                </div>
            )}

            {/* View Details Hint */}
            <div className="flex items-center gap-1 mt-2 text-xs text-gray-500 group-hover:text-primary-400 transition-colors">
                <span>Xem chi tiết</span>
                <svg
                    className="w-3 h-3"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 5l7 7-7 7"
                    />
                </svg>
            </div>
        </div>
    )
}
