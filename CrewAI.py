
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
import os

# =====================================
# 1. LOAD GEMINI API KEY
# =====================================

load_dotenv()

os.environ["GEMINI_API_KEY"]="GEMINI_API_KEY"

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Please configure it in your .env file."
    )

# =====================================
# 2. CONFIGURE GEMINI MODEL
# =====================================

gemini_llm = LLM(
    model="gemini/gemini-3.8-flash",
    api_key=gemini_api_key,
    temperature=0.3
)

# =====================================
# 3. CREATE AGENTS
# =====================================

researcher = Agent(
    role="Market Researcher",
    goal="Identify Tesla competitors and their electric vehicle models",
    backstory=(
        "You are an electric vehicle market research expert "
        "with knowledge of global automotive markets."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

analyst = Agent(
    role="Business Analyst",
    goal="Analyze competitors' strengths and weaknesses",
    backstory=(
        "You are an experienced automotive strategy consultant "
        "specializing in competitive analysis."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role="Report Writer",
    goal="Create a professional business report",
    backstory=(
        "You are an experienced business writer who prepares "
        "management-level strategic reports."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

# =====================================
# 4. CREATE TASK 1 - RESEARCH
# =====================================

research_task = Task(
    description=(
        "Identify Tesla's major competitors in the global electric "
        "vehicle market. List their companies, major EV models, "
        "market segments and key markets. "
        "Distinguish direct competitors from indirect competitors. "
        "Do not invent market statistics."
    ),
    expected_output=(
        "A structured list of Tesla competitors, their EV models, "
        "market segments and geographic presence."
    ),
    agent=researcher
)

# =====================================
# 5. CREATE TASK 2 - ANALYSIS
# =====================================

analyst_task = Task(
    description=(
        "Analyze the competitors identified by the researcher. "
        "Compare their strengths, weaknesses, pricing, technology, "
        "battery capabilities, charging infrastructure and brand position. "
        "Use the research task as your input. "
        "Clearly identify information that cannot be verified."
    ),
    expected_output=(
        "A competitor comparison including strengths, weaknesses "
        "and competitive positioning."
    ),
    agent=analyst,
    context=[research_task]
)

# =====================================
# 6. CREATE TASK 3 - FINAL REPORT
# =====================================

writer_task = Task(
    description=(
        "Prepare a professional management-level business report "
        "on Tesla's competitors. Include an executive summary, "
        "competitor overview, comparison, strengths, weaknesses, "
        "conclusions and research limitations."
    ),
    expected_output=(
        "A structured business report with clear headings, "
        "comparison tables and management recommendations."
    ),
    agent=writer,
    context=[research_task, analyst_task],
    output_file="Tesla_Competitor_Report.md"
)

# =====================================
# 7. CREATE CREW
# =====================================

crew = Crew(
    agents=[
        researcher,
        analyst,
        writer
    ],
    tasks=[
        research_task,
        analyst_task,
        writer_task
    ],
    process=Process.sequential,
    verbose=True
)

# =====================================
# 8. EXECUTE CREW
# =====================================

if __name__ == "__main__":

    print("\nStarting Tesla Competitor Research...\n")

    try:
        result = crew.kickoff()

        print("\n========== FINAL REPORT ==========\n")

        print(result)

        print("\nReport saved as Tesla_Competitor_Report.md")

    except Exception as error:
        print("\nCrewAI Execution Failed!")
        print("Error:", error)
