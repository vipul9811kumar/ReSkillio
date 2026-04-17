"""
Seed the ReSkillio market dataset with curated job descriptions.

Loads ~40 representative JDs across 8 industries and seniority levels,
runs them through the JD pipeline, stores results in BigQuery, then
rebuilds the industry_profiles table.

    python3 scripts/seed_market_data.py

Safe to re-run — each run generates new jd_ids so existing data is
preserved. Use --reset to truncate jd_profiles + jd_skills first.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from reskillio.models.jd import Industry
from reskillio.pipelines.jd_pipeline import run_jd_pipeline
from reskillio.storage.jd_store import JDStore

# ---------------------------------------------------------------------------
# Curated JD dataset
# Each entry: (title, company, industry, text)
# Mix of seniority levels and role types per industry — skills are explicit
# ---------------------------------------------------------------------------

CURATED_JDS = [

    # --- DATA & AI ---
    (
        "Senior Machine Learning Engineer", "NeuralWorks AI", Industry.DATA_AI,
        """We are looking for a Senior Machine Learning Engineer to join our AI platform team.
Requirements:
5+ years of experience in machine learning and Python. Deep expertise in TensorFlow, PyTorch,
and scikit-learn. Proficiency in MLOps practices including model versioning, feature engineering,
and monitoring. Experience with Vertex AI or SageMaker for model deployment. Strong SQL and
BigQuery skills for data analysis. Kubernetes and Docker for containerised model serving.
Nice to have:
Experience with LangChain or LlamaIndex for LLM applications. Knowledge of Spark or Kafka
for streaming ML pipelines. Published research or Kaggle competition experience."""
    ),
    (
        "Data Scientist", "Insightful Analytics", Industry.DATA_AI,
        """Join our data science team to build predictive models for customer behaviour.
Requirements:
3+ years in data science. Proficiency in Python, pandas, numpy, and scikit-learn.
Strong statistical modelling and machine learning fundamentals. SQL and experience with
BigQuery or Redshift. Excellent communication and stakeholder management skills.
Preferred:
Experience with NLP or computer vision. Familiarity with Airflow for pipeline orchestration.
Knowledge of A/B testing and experimentation frameworks."""
    ),
    (
        "Junior Data Scientist", "DataStart", Industry.DATA_AI,
        """Entry-level data scientist role for recent graduates.
Requirements:
1-2 years experience or strong academic background. Python, pandas, and matplotlib.
Understanding of machine learning concepts and scikit-learn. SQL for data querying.
Good communication and problem solving skills.
Preferred:
Experience with deep learning frameworks like TensorFlow or PyTorch.
Familiarity with cloud platforms like GCP or AWS."""
    ),
    (
        "MLOps Engineer", "ScaleML", Industry.DATA_AI,
        """Own the infrastructure behind our machine learning platform.
Requirements:
4+ years in MLOps or platform engineering. Kubernetes, Docker, Terraform for infrastructure.
Experience with Vertex AI, MLflow, or Kubeflow. Python scripting for automation.
CI/CD pipelines with GitHub Actions or Jenkins. Strong understanding of machine learning
workflows and model lifecycle management.
Nice to have:
Spark or Kafka for large-scale data processing. Experience with feature stores."""
    ),
    (
        "NLP Research Engineer", "LinguaAI", Industry.DATA_AI,
        """Build cutting-edge NLP systems for document understanding and generation.
Requirements:
Strong background in natural language processing and large language models.
Python, PyTorch, and Hugging Face transformers. Experience fine-tuning LLMs.
LangChain for orchestration. Deep understanding of attention mechanisms and transfer learning.
Preferred:
Published NLP research. Experience with RLHF or instruction tuning. Knowledge of
vector databases and RAG architectures."""
    ),

    # --- SOFTWARE ENGINEERING ---
    (
        "Senior Backend Engineer", "CloudStack Corp", Industry.SOFTWARE_ENGINEERING,
        """Build scalable backend services for our SaaS platform.
Requirements:
5+ years in backend engineering. Python or Go for service development. FastAPI or Django
for REST API design. PostgreSQL and Redis for data storage. Microservices architecture
and Docker/Kubernetes for deployment. GitHub Actions for CI/CD. Strong problem solving skills.
Nice to have:
GraphQL API experience. Kafka or RabbitMQ for event-driven systems. Elasticsearch."""
    ),
    (
        "Full Stack Engineer", "AppForge", Industry.SOFTWARE_ENGINEERING,
        """Develop and maintain our full stack web application.
Requirements:
3+ years full stack experience. React and TypeScript for frontend. Python with FastAPI
or Node.js for backend. PostgreSQL and REST API design. Docker for containerisation.
Git and agile development practices. Strong collaboration and communication.
Preferred:
Next.js for server-side rendering. Redis caching. Experience with cloud platforms."""
    ),
    (
        "Junior Software Engineer", "TechBridge", Industry.SOFTWARE_ENGINEERING,
        """We are hiring junior engineers eager to grow in a fast-paced environment.
Requirements:
0-2 years of experience. Python or JavaScript fundamentals. Basic SQL knowledge.
Understanding of REST APIs and Git version control. Good problem solving and teamwork.
Nice to have:
React or Vue.js experience. Docker basics. Exposure to agile or scrum."""
    ),
    (
        "Staff Software Engineer", "PlatformX", Industry.SOFTWARE_ENGINEERING,
        """Lead architectural decisions for our core platform.
Requirements:
8+ years in software engineering. Deep expertise in Python or Go with distributed systems.
Microservices, Kubernetes, Docker. PostgreSQL and NoSQL databases. Mentoring and leadership
experience. Strong stakeholder management and communication.
Preferred:
Rust or Scala for performance-critical systems. Experience with multi-region deployments."""
    ),
    (
        "Engineering Manager", "GrowthEng", Industry.SOFTWARE_ENGINEERING,
        """Lead a team of 6–8 engineers building our core product.
Requirements:
Experience managing engineering teams. Strong Python or Java background. Agile and scrum
for team delivery. Excellent communication, mentoring, and coaching skills.
Project management and stakeholder management. Ability to drive technical roadmap.
Nice to have:
Experience scaling teams from startup to growth stage. Background in platform or infrastructure."""
    ),

    # --- FINTECH ---
    (
        "Senior Data Engineer", "PayStream", Industry.FINTECH,
        """Build our real-time payments data platform.
Requirements:
5+ years data engineering. Python and SQL for pipeline development. Apache Spark and Kafka
for real-time processing. Airflow for orchestration. BigQuery or Redshift for analytics.
Experience in financial services or payments domain. PostgreSQL and dbt. Kubernetes and Docker.
Nice to have:
Knowledge of PCI-DSS compliance. Experience with fraud detection models. Scala for Spark."""
    ),
    (
        "Backend Engineer", "LendTech", Industry.FINTECH,
        """Build backend services for our lending and credit platform.
Requirements:
3+ years backend development. Python with FastAPI or Django. PostgreSQL for transactional
data. REST API design and microservices. Risk model integration experience. Docker and
Kubernetes. Understanding of compliance and data security in financial services.
Preferred:
Experience with open banking APIs. Redis for caching. Knowledge of credit scoring systems."""
    ),
    (
        "Quantitative Developer", "AlphaEdge Capital", Industry.FINTECH,
        """Develop quantitative trading systems and risk models.
Requirements:
Python and SQL for quantitative analysis. Statistical modelling and pandas, numpy.
Understanding of financial markets and trading systems. PostgreSQL for trade data storage.
Machine learning for predictive modelling. Strong problem solving and critical thinking.
Nice to have:
C++ for low-latency systems. Kafka for real-time market data. Experience with Bloomberg API."""
    ),
    (
        "Compliance Engineer", "RegShield", Industry.FINTECH,
        """Build tooling to ensure regulatory compliance across our platform.
Requirements:
3+ years engineering. Python and SQL. Understanding of financial regulations and compliance.
Data governance and audit logging. PostgreSQL and Elasticsearch for audit trails.
REST API development. Communication and stakeholder management.
Preferred:
Experience with AML or KYC systems. Knowledge of GDPR or PCI-DSS. Docker and Kubernetes."""
    ),
    (
        "Junior Data Analyst", "FinBase", Industry.FINTECH,
        """Support our analytics team with data-driven insights.
Requirements:
1-2 years experience. SQL and Python for data analysis. pandas and matplotlib for reporting.
Understanding of financial metrics and KPIs. Communication and presentation skills.
Preferred:
Experience with Tableau or Looker. Familiarity with BigQuery. Basic machine learning knowledge."""
    ),

    # --- HEALTHTECH ---
    (
        "Senior Software Engineer", "MedCore Platform", Industry.HEALTHTECH,
        """Build secure, HIPAA-compliant backend services for clinical workflows.
Requirements:
5+ years software engineering. Python and PostgreSQL. REST API design and microservices.
HIPAA compliance and healthcare data standards. Docker and Kubernetes. HL7 or FHIR experience
for health data interoperability. Strong problem solving and communication.
Nice to have:
Experience with EHR systems. ML models for clinical decision support. AWS or GCP."""
    ),
    (
        "Data Engineer", "HealthIQ", Industry.HEALTHTECH,
        """Process and transform clinical and claims data at scale.
Requirements:
3+ years data engineering. Python and SQL. Apache Spark for large-scale processing.
Airflow for workflow orchestration. PostgreSQL and BigQuery. HIPAA awareness.
Data quality and governance experience. Docker.
Preferred:
Experience with clinical data standards like ICD-10, CPT codes. dbt for transformations."""
    ),
    (
        "Healthcare Data Scientist", "ClinicalML", Industry.HEALTHTECH,
        """Apply machine learning to clinical trial and patient outcome data.
Requirements:
Python, pandas, scikit-learn, and statistical modelling. SQL and BigQuery. Understanding
of clinical data and research methodologies. HIPAA compliance. Communication and
presentation skills. Machine learning and data analysis.
Nice to have:
R for biostatistics. Deep learning for medical imaging. Experience with survival analysis."""
    ),
    (
        "Product Engineer", "PatientFirst", Industry.HEALTHTECH,
        """Build patient-facing web and mobile features for our care platform.
Requirements:
3+ years engineering. React and TypeScript. Python or Node.js backend. REST API design.
PostgreSQL. HIPAA-compliant development practices. Agile development and collaboration.
Nice to have:
React Native for mobile. Understanding of healthcare workflows. Experience with FHIR APIs."""
    ),
    (
        "Junior Backend Engineer", "WellnessStack", Industry.HEALTHTECH,
        """Entry-level backend role in a fast-growing digital health startup.
Requirements:
0-2 years. Python basics and REST API fundamentals. SQL for data queries. Git version control.
Understanding of software development lifecycle. Teamwork and communication.
Preferred:
Django or FastAPI. Docker basics. Interest in healthcare technology."""
    ),

    # --- ECOMMERCE ---
    (
        "Senior Platform Engineer", "ShopVault", Industry.ECOMMERCE,
        """Scale our ecommerce infrastructure to handle peak retail traffic.
Requirements:
5+ years platform engineering. Python and Go. Kubernetes and Docker for container orchestration.
Redis for caching. PostgreSQL and Elasticsearch for product search. Kafka for event streaming.
Microservices architecture. GitHub Actions for CI/CD. Strong problem solving skills.
Nice to have:
Experience with Shopify or Magento integration. GraphQL APIs. Terraform for IaC."""
    ),
    (
        "Data Engineer", "RetailMetrics", Industry.ECOMMERCE,
        """Build data pipelines powering our personalisation and recommendation engine.
Requirements:
Python, SQL, Apache Spark. Airflow for orchestration. BigQuery or Redshift. dbt for
data modelling. Kafka for real-time event processing. PostgreSQL. Docker and Kubernetes.
Preferred:
Experience with recommendation systems. Machine learning with scikit-learn or PyTorch.
Understanding of ecommerce metrics like conversion rate, LTV, AOV."""
    ),
    (
        "Backend Engineer", "CartPro", Industry.ECOMMERCE,
        """Build the checkout and payments backend for millions of daily orders.
Requirements:
3+ years. Python with FastAPI. PostgreSQL and Redis. REST API and microservices design.
Docker. Experience with payment gateways. Understanding of supply chain or fulfilment.
Agile and scrum. Strong collaboration.
Nice to have:
Kafka for order event streaming. Kubernetes. Elasticsearch for order search."""
    ),
    (
        "Machine Learning Engineer", "Personalise.io", Industry.ECOMMERCE,
        """Build real-time recommendation and personalisation models.
Requirements:
Python, scikit-learn, TensorFlow or PyTorch. SQL and BigQuery. Feature engineering and
machine learning pipelines. Spark for large-scale data processing. Kubernetes for model serving.
Communication and stakeholder management.
Preferred:
Experience with collaborative filtering or content-based recommendation. Kafka for
real-time feature serving. A/B testing frameworks."""
    ),
    (
        "Junior Data Analyst", "TrendCart", Industry.ECOMMERCE,
        """Analyse sales and customer behaviour to support merchandising decisions.
Requirements:
SQL and Python basics. pandas and matplotlib. Understanding of ecommerce KPIs.
Communication and presentation skills. Data analysis and critical thinking.
Preferred:
Experience with Tableau or Looker. BigQuery or Redshift. Knowledge of A/B testing."""
    ),

    # --- CYBERSECURITY ---
    (
        "Senior Security Engineer", "ShieldNet", Industry.CYBERSECURITY,
        """Design and implement security controls across our cloud infrastructure.
Requirements:
5+ years in cybersecurity. Python for security automation and tooling. CISSP or CISA
certification preferred. Cloud security on AWS or GCP. SIEM platforms and threat detection.
Zero trust network architecture. Vulnerability assessment and penetration testing.
Firewall and network security. Strong communication and stakeholder management.
Nice to have:
CISSP, CEH, or cloud security certifications. Kubernetes security hardening. IDS/IPS."""
    ),
    (
        "Security Analyst", "CyberGuard", Industry.CYBERSECURITY,
        """Monitor and respond to security incidents in our SOC.
Requirements:
3+ years in security operations. SIEM tools for threat detection and incident response.
Network security and firewall management. Vulnerability scanning and risk assessment.
Python for automation. Understanding of compliance frameworks (ISO27001, SOC2).
Communication and critical thinking.
Nice to have:
Penetration testing experience. CySA+ or Security+ certification. Cloud security."""
    ),
    (
        "Application Security Engineer", "SecureCode", Industry.CYBERSECURITY,
        """Embed security into our software development lifecycle.
Requirements:
Python and experience with secure coding practices. OWASP top 10 and secure code review.
Static and dynamic analysis tools (SAST, DAST). DevSecOps and CI/CD security integration.
REST API security. Penetration testing. Collaboration and communication.
Preferred:
Experience with GitHub Advanced Security. Container security scanning. Kubernetes RBAC."""
    ),
    (
        "Cloud Security Architect", "NimbusShield", Industry.CYBERSECURITY,
        """Lead the security architecture for our multi-cloud environment.
Requirements:
8+ years in security with focus on cloud. AWS and GCP security services. Terraform for
secure infrastructure-as-code. Zero trust and identity management. CISSP or CCSP certification.
Kubernetes security. Strong leadership and stakeholder management.
Nice to have:
Experience with CSPM tools. eBPF-based security monitoring. SOC2 or ISO27001 audit experience."""
    ),
    (
        "Junior Security Analyst", "DefendBase", Industry.CYBERSECURITY,
        """Join our SOC team as a junior analyst — great opportunity for security graduates.
Requirements:
0-2 years. Security+ or CEH certification a plus. Understanding of network security basics.
SIEM log analysis. Python scripting fundamentals. Communication and teamwork.
Preferred:
Hands-on CTF experience. Familiarity with Splunk or Microsoft Sentinel. Linux proficiency."""
    ),

    # --- CLOUD / DEVOPS ---
    (
        "Senior DevOps Engineer", "InfraFlow", Industry.CLOUD_DEVOPS,
        """Own the cloud infrastructure and CI/CD pipeline for our platform.
Requirements:
5+ years DevOps or platform engineering. Kubernetes and Docker for container orchestration.
Terraform and Ansible for infrastructure-as-code. GCP, AWS, or Azure. GitHub Actions and
Jenkins for CI/CD. Python for automation. Prometheus and Grafana for monitoring.
Microservices and cloud-native architecture. Strong problem solving.
Nice to have:
Helm for Kubernetes packaging. Vault for secrets management. Datadog or Dynatrace."""
    ),
    (
        "Site Reliability Engineer", "UpTime Labs", Industry.CLOUD_DEVOPS,
        """Ensure reliability and performance of our distributed systems.
Requirements:
4+ years SRE or platform engineering. Kubernetes and Docker. Python or Go for tooling.
Prometheus, Grafana and alerting. GCP or AWS. Terraform for infrastructure management.
CI/CD with GitHub Actions. Strong incident management and on-call experience.
Preferred:
Chaos engineering experience. Service mesh (Istio). Experience with multi-region failover."""
    ),
    (
        "Cloud Infrastructure Engineer", "CloudBase", Industry.CLOUD_DEVOPS,
        """Build and manage our cloud infrastructure on GCP.
Requirements:
3+ years cloud engineering. GCP services including Cloud Run, GKE, BigQuery. Terraform.
Docker and Kubernetes. Python scripting for automation. GitHub Actions for CI/CD.
Networking and security fundamentals. Strong collaboration.
Nice to have:
Multi-cloud experience (AWS, Azure). Ansible for configuration management. Helm."""
    ),
    (
        "Platform Engineer", "DevXcel", Industry.CLOUD_DEVOPS,
        """Build the internal developer platform to accelerate engineering velocity.
Requirements:
Kubernetes and Docker expertise. Terraform for infrastructure-as-code. GitHub Actions and
ArgoCD for GitOps. Python for tooling. AWS or GCP. Strong communication and collaboration.
Mentoring junior engineers. Agile development practices.
Preferred:
Experience with Backstage.io or internal developer portals. Crossplane. Vault for secrets."""
    ),
    (
        "Junior DevOps Engineer", "SpinUp", Industry.CLOUD_DEVOPS,
        """Entry-level DevOps role — learn cloud-native infrastructure from the ground up.
Requirements:
0-2 years. Basic Linux and shell scripting. Docker fundamentals. Git and CI/CD concepts.
Understanding of cloud platforms (GCP, AWS, or Azure). Communication and teamwork.
Nice to have:
Kubernetes basics. Terraform introduction. Python scripting. Any cloud certification."""
    ),

    # --- PRODUCT MANAGEMENT ---
    (
        "Senior Product Manager", "ProductHQ", Industry.PRODUCT_MANAGEMENT,
        """Lead product strategy for our core B2B SaaS platform.
Requirements:
5+ years product management. Strong agile and scrum expertise. Stakeholder management
and executive communication. Data-driven decision making with SQL and analytics tools.
Product strategy and roadmap planning. User research and customer interviews.
Experience working with engineering and design teams. OKR framework.
Nice to have:
Technical background in software engineering. Experience with product-led growth.
Familiarity with Figma or Miro for prototyping."""
    ),
    (
        "Product Manager", "GrowthLoop", Industry.PRODUCT_MANAGEMENT,
        """Own the roadmap for our growth and monetisation products.
Requirements:
3+ years product management. Agile development and scrum. SQL for product analytics.
Stakeholder management and communication. A/B testing and experimentation. User research.
Go-to-market planning. Strong critical thinking and problem solving.
Preferred:
Experience with growth or monetisation products. Familiarity with Amplitude or Mixpanel.
Background in ecommerce or SaaS."""
    ),
    (
        "Technical Product Manager", "DevTools Co", Industry.PRODUCT_MANAGEMENT,
        """Drive the product roadmap for our developer tools platform.
Requirements:
Product management experience with a strong technical background. Understanding of
REST APIs, microservices, and software development lifecycle. SQL for data analysis.
Agile and scrum. Stakeholder management and engineering collaboration.
Documentation and presentation skills.
Nice to have:
Previous software engineering experience. Experience with API products or developer platforms.
Knowledge of CI/CD and DevOps workflows."""
    ),
    (
        "Associate Product Manager", "StartPM", Industry.PRODUCT_MANAGEMENT,
        """Great entry point for aspiring product managers.
Requirements:
1-2 years in product management or a technical role. Agile fundamentals. SQL basics
for data analysis. Communication and presentation skills. User empathy and problem solving.
Collaboration with engineering and design teams.
Preferred:
Experience with product analytics tools. Background in engineering or UX research.
Familiarity with product frameworks like JTBD or double diamond."""
    ),
    (
        "Head of Product", "ScaleUp Ventures", Industry.PRODUCT_MANAGEMENT,
        """Define and own the product vision across our portfolio of products.
Requirements:
8+ years product management, including leadership. Product strategy and OKR frameworks.
Stakeholder management at C-level. Agile at scale. Data-driven approach with SQL and analytics.
Mentoring and coaching product teams. Go-to-market execution. Strong communication and leadership.
Nice to have:
Experience in a high-growth startup. Background in machine learning or data products.
Familiarity with platform business models."""
    ),
]


def main(reset: bool = False) -> None:
    if not settings.gcp_project_id:
        logger.error("GCP_PROJECT_ID not set")
        sys.exit(1)

    import os
    if settings.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials

    store = JDStore(project_id=settings.gcp_project_id)
    store.ensure_tables()

    if reset:
        logger.warning("Resetting jd_profiles and jd_skills tables...")
        for table in [store.jd_profiles_table, store.jd_skills_table]:
            store.client.query(f"DELETE FROM `{table}` WHERE TRUE").result()
        logger.info("Tables cleared.")

    logger.info(f"Seeding {len(CURATED_JDS)} job descriptions...")
    success = 0
    for i, (title, company, industry, text) in enumerate(CURATED_JDS, 1):
        try:
            result = run_jd_pipeline(
                text=text,
                title=title,
                company=company,
                industry=industry,
                model_name="en_core_web_lg",
                store=store,
            )
            logger.info(
                f"  [{i:02d}/{len(CURATED_JDS)}] {title[:45]:45} "
                f"| {industry.value:22} | seniority={result.seniority.value:8} "
                f"| req={len(result.required_skills):2} pref={len(result.preferred_skills):2}"
            )
            success += 1
        except Exception as exc:
            logger.error(f"  [{i:02d}] FAILED {title}: {exc}")

    logger.info(f"\nSeeded {success}/{len(CURATED_JDS)} JDs successfully.")

    # Derive industry profiles
    logger.info("Building industry_profiles table...")
    store.refresh_industry_profiles()
    logger.info("industry_profiles ready.")

    # Summary per industry
    logger.info("\nTop 5 skills per industry:")
    profiles = store.get_all_industry_profiles()
    for industry_name, skills in sorted(profiles.items()):
        top5 = [s["skill_name"] for s in skills[:5]]
        logger.info(f"  {industry_name:25} → {top5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Truncate tables before seeding")
    args = parser.parse_args()
    main(reset=args.reset)
