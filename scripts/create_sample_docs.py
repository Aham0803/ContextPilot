"""
Script to generate sample PDF documents with multi-page structures,
departments, categories, and deliberate edge-cases (e.g. empty pages).
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def create_sample_pdfs(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2563eb'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Italic'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=14
    )

    # 1. Employee Handbook 2024
    pdf_path_1 = os.path.join(output_dir, "employee_handbook_2024.pdf")
    doc1 = SimpleDocTemplate(pdf_path_1, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story1 = []

    # Page 1
    story1.append(Paragraph("Enterprise Employee Handbook (2024 Edition)", title_style))
    story1.append(Paragraph("Department: Human Resources | Category: Policy | Effective Year: 2024", meta_style))
    story1.append(Paragraph("Section 1: Working Hours and Remote Work Arrangements", h2_style))
    story1.append(Paragraph(
        "Our standard working hours span 9:00 AM to 5:00 PM in the employee's local time zone, with core collaboration hours "
        "designated between 10:00 AM and 4:00 PM. During core hours, team members are expected to be available for team syncs and cross-functional communication.",
        body_style
    ))
    story1.append(Paragraph(
        "Employees operating in a hybrid configuration are required to work at least 2 designated days per week from a regional office. "
        "Full-time remote team members are eligible for a one-time home office setup reimbursement of $1,000 and an ongoing annual remote work stipend of $500.",
        body_style
    ))
    story1.append(PageBreak())

    # Page 2
    story1.append(Paragraph("Section 2: Paid Time Off (PTO) and Sick Leave", h2_style))
    story1.append(Paragraph(
        "Full-time regular employees accrue 20 days of Paid Time Off (PTO) per calendar year, accrued proportionally on a monthly schedule. "
        "A maximum of 5 unused PTO days may be rolled over into the subsequent calendar year, expiring on March 31st.",
        body_style
    ))
    story1.append(Paragraph(
        "Employees are granted 10 dedicated sick and wellness days annually. In addition, eligible primary caregivers are entitled to 16 weeks of fully paid "
        "parental leave following the birth, adoption, or foster placement of a child. Secondary caregivers receive 8 weeks of fully paid leave.",
        body_style
    ))
    story1.append(PageBreak())

    # Page 3
    story1.append(Paragraph("Section 3: Professional Development and Travel Stipends", h2_style))
    story1.append(Paragraph(
        "Continuous learning is vital to our organization. Each permanent employee is allocated an annual professional development budget of $2,500. "
        "This budget can be utilized for industry certifications, conferences, educational textbooks, and technical workshops.",
        body_style
    ))
    story1.append(Paragraph(
        "Corporate standard hardware includes modern workstation laptops refreshed on a strict 36-month cycle. "
        "For business travel, corporate cards must be used, with an approved standard per diem rate of $75 per day for meals and incidentals.",
        body_style
    ))
    story1.append(PageBreak())

    # Page 4: Empty page (for testing empty page ingestion handling)
    story1.append(Spacer(1, 100))
    doc1.build(story1)
    print(f"Created: {pdf_path_1}")

    # 2. Cloud Architecture Guide
    pdf_path_2 = os.path.join(output_dir, "cloud_architecture_guide.pdf")
    doc2 = SimpleDocTemplate(pdf_path_2, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story2 = []

    # Page 1
    story2.append(Paragraph("Cloud Infrastructure & Architecture Blueprint", title_style))
    story2.append(Paragraph("Department: Engineering | Category: Technical | Published Year: 2023", meta_style))
    story2.append(Paragraph("Module 1: Microservices Architecture and Service Mesh", h2_style))
    story2.append(Paragraph(
        "The backend architecture comprises decoupled containerized microservices hosted on multi-tenant Kubernetes clusters. "
        "Ingress routing and traffic routing are managed through an Envoy-based API Gateway enforcing strict rate limiting of 10,000 requests per second per tenant.",
        body_style
    ))
    story2.append(Paragraph(
        "Service-to-service communication is orchestrated via an Istio service mesh, providing mutual TLS (mTLS) encryption by default. "
        "All ingress API endpoints require OAuth 2.0 JWT bearer tokens validated against our centralized identity provider.",
        body_style
    ))
    story2.append(PageBreak())

    # Page 2
    story2.append(Paragraph("Module 2: Vector Database and Semantic Search Tier", h2_style))
    story2.append(Paragraph(
        "A vector database is an optimized storage and retrieval engine that indexes high-dimensional vector embeddings generated by machine learning models. "
        "Unlike traditional relational databases that match exact keywords or scalar attributes, a vector database performs approximate nearest neighbor (ANN) search "
        "to discover semantically similar passages based on cosine distance or dot-product similarity.",
        body_style
    ))
    story2.append(Paragraph(
        "Our RAG search tier leverages HNSW (Hierarchical Navigable Small World) indexing to maintain sub-50 millisecond retrieval latency across millions of chunk vectors. "
        "A distributed Redis cache layer acts as a semantic cache with a cosine similarity threshold of 0.92, bypassing vector index computation for repeated queries.",
        body_style
    ))
    story2.append(PageBreak())

    # Page 3
    story2.append(Paragraph("Module 3: High Availability and Disaster Recovery (DR)", h2_style))
    story2.append(Paragraph(
        "To ensure business continuity, our cloud services run in an active-active multi-region configuration across US-East and US-West regions. "
        "Database replication utilizes synchronous multi-AZ replicas with asynchronous cross-region streaming.",
        body_style
    ))
    story2.append(Paragraph(
        "Our Disaster Recovery Service Level Agreements (SLAs) enforce a Recovery Point Objective (RPO) of under 5 minutes and a "
        "Recovery Time Objective (RTO) of under 15 minutes. Automated failover health checks trigger DNS reconfiguration via Route53 when 3 consecutive probes fail.",
        body_style
    ))
    doc2.build(story2)
    print(f"Created: {pdf_path_2}")

    # 3. AI Governance Policy
    pdf_path_3 = os.path.join(output_dir, "ai_governance_policy.pdf")
    doc3 = SimpleDocTemplate(pdf_path_3, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story3 = []

    # Page 1
    story3.append(Paragraph("Enterprise AI Governance & Ethics Policy", title_style))
    story3.append(Paragraph("Department: Compliance | Category: Governance | Enacted Year: 2024", meta_style))
    story3.append(Paragraph("Article 1: Core AI Principles and Human-in-the-Loop Oversight", h2_style))
    story3.append(Paragraph(
        "All generative AI and machine learning systems deployed within enterprise operations must adhere to core ethical pillars: "
        "algorithmic fairness, transparency, auditability, data privacy, and human oversight. Autonomous decision-making is strictly prohibited "
        "in high-stakes scenarios such as employment screening, credit underwriting, and termination decisions.",
        body_style
    ))
    story3.append(Paragraph(
        "A mandatory human-in-the-loop (HITL) review is required for any automated output that directly affects client financial status or employee welfare.",
        body_style
    ))
    story3.append(PageBreak())

    # Page 2
    story3.append(Paragraph("Article 2: Data Protection, PII Redaction, and Training Restrictions", h2_style))
    story3.append(Paragraph(
        "Confidential corporate data, proprietary source code, and customer Personally Identifiable Information (PII) must be scrubbed using automated "
        "regex and NER token maskers prior to embedding generation or prompt transmission to third-party Large Language Model APIs.",
        body_style
    ))
    story3.append(Paragraph(
        "Model inference logs and intermediate retrieval contexts must not be retained beyond a strict 90-day retention window. "
        "Enterprise customer data shall never be utilized to fine-tune external foundation models without explicit written bilateral consent.",
        body_style
    ))
    story3.append(PageBreak())

    # Page 3
    story3.append(Paragraph("Article 3: Evaluation Standards and Incident Reporting", h2_style))
    story3.append(Paragraph(
        "AI pipelines undergo mandatory quarterly adversarial red-teaming. For Retrieval-Augmented Generation systems, hallucination rates must remain "
        "below a strict threshold of 2.0% on domain evaluation benchmark suites before production rollout.",
        body_style
    ))
    story3.append(Paragraph(
        "Any discovered hallucination or security vulnerability categorized as Severity 1 (such as prompt injection exfiltrating PII) must be escalated "
        "to the AI Safety Committee and CISO within 24 hours of identification.",
        body_style
    ))
    doc3.build(story3)
    print(f"Created: {pdf_path_3}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_documents")
    create_sample_pdfs(out_dir)
