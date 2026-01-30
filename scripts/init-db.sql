-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create candidates table
CREATE TABLE IF NOT EXISTS candidates (
    id VARCHAR(36) PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    date_of_birth DATE,
    address TEXT,
    summary TEXT,
    raw_resume JSONB,
    summary_embedding vector(1024),
    total_experience_years FLOAT,
    validation_warnings JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id VARCHAR(36) PRIMARY KEY,
    candidate_id VARCHAR(36) REFERENCES candidates(id) ON DELETE CASCADE,
    parent_id VARCHAR(36),
    section VARCHAR(100),
    subsection VARCHAR(255),
    content TEXT,
    enriched_content TEXT,
    embedding vector(1024),
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_candidates_email ON candidates(email);
CREATE INDEX IF NOT EXISTS idx_candidates_name ON candidates(full_name);
CREATE INDEX IF NOT EXISTS idx_chunks_candidate ON chunks(candidate_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);

-- Create vector indexes for similarity search
CREATE INDEX IF NOT EXISTS idx_candidates_summary_embedding 
    ON candidates USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for candidates table
DROP TRIGGER IF EXISTS update_candidates_updated_at ON candidates;
CREATE TRIGGER update_candidates_updated_at
    BEFORE UPDATE ON candidates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
