-- Add new columns to topics table
ALTER TABLE topics 
ADD COLUMN IF NOT EXISTS label_confidence FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS keywords TEXT[] DEFAULT '{}',
ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'seedling',
ADD COLUMN IF NOT EXISTS alias VARCHAR(50)[],
ADD COLUMN IF NOT EXISTS blurb TEXT;

-- Create index for status field for faster filtering
CREATE INDEX IF NOT EXISTS idx_topics_status ON topics(status);

-- Create GIN index for array operations on keywords
CREATE INDEX IF NOT EXISTS idx_topics_keywords ON topics USING GIN(keywords);

-- Create GIN index for array operations on alias
CREATE INDEX IF NOT EXISTS idx_topics_alias ON topics USING GIN(alias);
