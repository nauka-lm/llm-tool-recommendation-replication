-- ============================================================
-- CadcomOnline Database Schema
-- Engineering Tool Inventory for Anti-Hallucination Evaluation
-- ============================================================
-- This schema supports the evaluation framework described in:
-- "Hallucination Mitigation in LLM-Based Tool Recommendation:
--  A Cross-Provider Architectural Ablation Study"
-- ============================================================

-- Panels (top-level classification)
CREATE TABLE IF NOT EXISTS "Panels" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(250) NOT NULL,
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedBy" VARCHAR(100),
    "CreatedDate" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "LastModifiedBy" VARCHAR(100),
    "LastModifiedDate" TIMESTAMP WITH TIME ZONE
);

-- Categories (second-level classification)
CREATE TABLE IF NOT EXISTS "Categories" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(250) NOT NULL,
    "PanelId" INTEGER NOT NULL REFERENCES "Panels"("Id"),
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedBy" VARCHAR(100),
    "CreatedDate" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "LastModifiedBy" VARCHAR(100),
    "LastModifiedDate" TIMESTAMP WITH TIME ZONE
);

-- Subcategories (third-level, maps to evaluation domains D1-D4)
CREATE TABLE IF NOT EXISTS "Subcategories" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(250) NOT NULL,
    "CategoryId" INTEGER NOT NULL REFERENCES "Categories"("Id"),
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedBy" VARCHAR(100),
    "CreatedDate" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "LastModifiedBy" VARCHAR(100),
    "LastModifiedDate" TIMESTAMP WITH TIME ZONE
);

-- Tool types
CREATE TABLE IF NOT EXISTS "TypeOfTools" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(50) NOT NULL,
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedBy" VARCHAR(100),
    "CreatedDate" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "LastModifiedBy" VARCHAR(100),
    "LastModifiedDate" TIMESTAMP WITH TIME ZONE
);

-- Tool Passports (the 82 engineering tools - core inventory)
CREATE TABLE IF NOT EXISTS "ToolPassports" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(250) NOT NULL,
    "SubcategoryId" INTEGER NOT NULL REFERENCES "Subcategories"("Id"),
    "CategoryId" INTEGER REFERENCES "Categories"("Id"),
    "PanelId" INTEGER REFERENCES "Panels"("Id"),
    "TypeOfToolId" INTEGER REFERENCES "TypeOfTools"("Id"),
    "Function" TEXT,
    "AbstractionLevel" TEXT,
    "IOConnection" TEXT,
    "VerificationTool" TEXT,
    "ApplicationAreaOrEquivalenceCoverage" TEXT,
    "QualitativeFeature" TEXT,
    "QuantitativeFeature" TEXT,
    "ToolURL" VARCHAR(500),
    "Logo" TEXT,
    "LogoName" VARCHAR(250),
    "ApprovalStatus" VARCHAR(20) DEFAULT 'Approved',
    "SubmittedByUserId" VARCHAR(100),
    "ReviewedBy" VARCHAR(100),
    "ReviewDate" TIMESTAMP WITH TIME ZONE,
    "ReviewComments" TEXT,
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedBy" VARCHAR(100),
    "CreatedDate" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "LastModifiedBy" VARCHAR(100),
    "LastModifiedDate" TIMESTAMP WITH TIME ZONE
);

-- Evaluation criteria for tools
CREATE TABLE IF NOT EXISTS "Criterias" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(100) NOT NULL,
    "SubcategoryId" INTEGER NOT NULL REFERENCES "Subcategories"("Id"),
    "IsMandatory" BOOLEAN NOT NULL DEFAULT FALSE,
    "WeightCoefficient" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    "ApprovalStatus" VARCHAR(20) DEFAULT 'Approved',
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedBy" VARCHAR(100),
    "CreatedDate" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "LastModifiedBy" VARCHAR(100),
    "LastModifiedDate" TIMESTAMP WITH TIME ZONE
);

-- Possible values for criteria
CREATE TABLE IF NOT EXISTS "CriteriaValues" (
    "Id" SERIAL PRIMARY KEY,
    "CriteriaId" INTEGER NOT NULL REFERENCES "Criterias"("Id"),
    "Value" VARCHAR(250) NOT NULL,
    "ApprovalStatus" VARCHAR(20) DEFAULT 'Approved',
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedBy" VARCHAR(100),
    "CreatedDate" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "LastModifiedBy" VARCHAR(100),
    "LastModifiedDate" TIMESTAMP WITH TIME ZONE
);

-- Junction table: tool features (links tools to criteria values)
CREATE TABLE IF NOT EXISTS "ToolFeatures" (
    "Id" SERIAL PRIMARY KEY,
    "ToolPassportId" INTEGER NOT NULL REFERENCES "ToolPassports"("Id"),
    "CriteriaId" INTEGER NOT NULL REFERENCES "Criterias"("Id"),
    "CriteriaValueId" INTEGER NOT NULL REFERENCES "CriteriaValues"("Id"),
    "SubcategoryId" INTEGER REFERENCES "Subcategories"("Id"),
    "CategoryId" INTEGER REFERENCES "Categories"("Id"),
    "PanelId" INTEGER REFERENCES "Panels"("Id"),
    "ApprovalStatus" VARCHAR(20) DEFAULT 'Approved',
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedBy" VARCHAR(100),
    "CreatedDate" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "LastModifiedBy" VARCHAR(100),
    "LastModifiedDate" TIMESTAMP WITH TIME ZONE
);

-- Performance indexes (used by DatabaseService.cs)
CREATE INDEX IF NOT EXISTS "IX_ToolFeature_SubcategoryId"
    ON "ToolFeatures" ("SubcategoryId", "IsDeleted", "IsActive");

CREATE INDEX IF NOT EXISTS "IX_ToolFeature_ToolPassportId"
    ON "ToolFeatures" ("ToolPassportId", "IsDeleted", "IsActive");

CREATE INDEX IF NOT EXISTS "IX_Criteria_SubcategoryId"
    ON "Criterias" ("SubcategoryId", "IsDeleted", "IsActive");

CREATE INDEX IF NOT EXISTS "IX_ToolPassport_SubcategoryId"
    ON "ToolPassports" ("SubcategoryId", "IsDeleted", "IsActive");

CREATE INDEX IF NOT EXISTS "IX_CriteriaValue_CriteriaId"
    ON "CriteriaValues" ("CriteriaId", "IsDeleted", "IsActive");

CREATE INDEX IF NOT EXISTS "IX_ToolPassport_ApprovalStatus"
    ON "ToolPassports" ("ApprovalStatus");

-- ============================================================
-- Domain-to-Subcategory Mapping (used by EvaluationRunner.cs)
-- ============================================================
-- D1 -> Subcategory.Name = "PCB Design Tool"
-- D2 -> Subcategory.Name = "PCB Design Calculator"
-- D3 -> Subcategory.Name = "Switched-mode power supplies (SMPS), converters, regulators"
-- D4 -> Subcategory.Name = "Transformers"
-- ============================================================

COMMENT ON TABLE "ToolPassports" IS
    'Core tool inventory table. Contains 82 verified engineering tools '
    'used as ground truth for hallucination detection. Each tool belongs '
    'to one subcategory (evaluation domain). The Name field is the '
    'canonical tool name used for grounding classification.';

COMMENT ON TABLE "Subcategories" IS
    'Maps to evaluation domains D1-D4. The Name field must match '
    'the domain definitions in EvaluationRunner.DomainDefinitions.';
