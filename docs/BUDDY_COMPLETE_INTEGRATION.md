# Buddy AI Adaptation Guide with MeTTa & OpenClaw Integration

**Purpose**: Complete integration roadmap showing how LARQL patterns, MeTTa logic, and OpenClaw agent patterns all flow into Buddy through SmartMan architecture.

**Core Philosophy**: Multiple donor systems → SmartMan authority → Buddy operator layer

---

## Integration Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DONOR SYSTEMS (Study Lane)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LARQL (Local Knowledge)    MeTTa (Logic Layer)    OpenClaw     │
│  ├─ Immutable base          ├─ Reasoning engine   ├─ Session   │
│  ├─ Patch overlays          ├─ Knowledge graphs   ├─ Tools     │
│  ├─ Walk-only mode          ├─ Inference chains   ├─ Memory    │
│  ├─ Template cache          ├─ Type system        ├─ Planning  │
│  └─ Profiling               └─ Proof tracking     └─ Refusal   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Extract patterns
┌─────────────────────────────────────────────────────────────────┐
│                    SMARTMAN (Authority Root)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Core Authority:                                                │
│  ├─ Immutable state.json (base truth)                          │
│  ├─ Receipt chain (append-only log)                            │
│  ├─ Approval gates (all mutations reviewed)                    │
│  └─ Compact state (current = base + receipts)                  │
│                                                                  │
│  Pattern Integration:                                           │
│  ├─ LARQL → runtime/state/compact_state.py                     │
│  ├─ MeTTa → runtime/brain/logic_engine.py                      │
│  └─ OpenClaw → runtime/session_manager.py                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Operates over
┌─────────────────────────────────────────────────────────────────┐
│                    BUDDY (Operator Layer)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User-Facing Interface:                                         │
│  ├─ Natural language commands                                   │
│  ├─ Tool execution (GitHub, GDrive, Browser)                   │
│  ├─ Posture awareness (witness/operator/authority)             │
│  └─ Receipt rendering (what happened, why)                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Visualizes in
┌─────────────────────────────────────────────────────────────────┐
│                    SOURCE 2 (Shell Layer)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  3D Workspace (Display Only):                                   │
│  ├─ Task cards (floating 3D objects)                           │
│  ├─ Tool palette (wall of available tools)                     │
│  ├─ Receipt wall (audit trail visualization)                   │
│  └─ Buddy avatar (character representation)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: LARQL → SmartMan Pattern Extraction

### Pattern 1: Immutable Base + Receipt Overlay

**LARQL Source**:
- Base vindex (readonly mmap'd files)
- Patches (.vlp JSON files, stackable)
- PatchedVindex (runtime overlay)
- COMPILE INTO VINDEX (bake patches into new base)

**SmartMan Landing**:
```python
# smartman/runtime/state/compact_state.py

from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import json

@dataclass
class Receipt:
    """Single state mutation - append-only log entry"""
    timestamp: str
    operation: str  # "set", "delete", "merge"
    path: str       # dot-notation: "tools.browser.headless"
    value: Any
    previous_value: Any
    hash: str
    approved_by: str  # "smartman", "user", "system"
    source: str       # Which donor pattern inspired this

class CompactState:
    """
    Pattern from LARQL: Immutable base + overlay.
    SmartMan adaptation: state.json never modified,
    all changes via receipts.jsonl
    """
    def __init__(self, base_path: Path):
        self.base_path = base_path / "state.json"
        self.receipts_path = base_path / "receipts.jsonl"

        # Load immutable base
        self._base = self._load_base()

        # Load receipt log
        self._receipts: List[Receipt] = self._load_receipts()

    def current_state(self) -> Dict:
        """
        Compute current state = base + all receipts.
        Pattern from LARQL: walk FFN computes on-demand.
        """
        state = self._base.copy()
        for receipt in self._receipts:
            state = self._apply_receipt(receipt, state)
        return state

    def propose_change(self, path: str, value: Any) -> Receipt:
        """
        Create receipt proposal (not yet approved).
        Pattern from LARQL: INSERT generates proposal first.
        """
        # Implementation here
        pass

    def compact_base(self, output_path: Path):
        """
        Bake current state into new base file.
        Pattern from LARQL: COMPILE CURRENT INTO VINDEX.
        """
        current = self.current_state()
        output_path.write_text(json.dumps(current, indent=2))
```

---

## Part 2: MeTTa → SmartMan Logic Integration

### MeTTa Donor Patterns

**What MeTTa Provides**:
- **Reasoning engine**: Logic-based inference
- **Knowledge graphs**: Structured relationships
- **Type system**: Strongly typed facts
- **Proof tracking**: Justify conclusions
- **Composable reasoning**: Chain inferences

**SmartMan Landing**:
```python
# smartman/runtime/brain/logic_engine.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Fact:
    """
    MeTTa-inspired fact with type and confidence.
    Pattern: Every fact has provenance + confidence score.
    """
    subject: str
    predicate: str
    object: str
    confidence: float  # 0.0 to 1.0
    source: str        # Where it came from
    proof_chain: List[str]  # How we derived it

class LogicEngine:
    """
    MeTTa-inspired reasoning layer for Buddy.
    Handles: inference chains, knowledge graphs, proof tracking.
    """

    def __init__(self):
        self.facts: Dict[str, Fact] = {}
        self.rules: List[InferenceRule] = []

    def add_fact(self, fact: Fact, approved_by: str = "pending"):
        """
        Add fact to knowledge base.
        Pattern from MeTTa: Facts are typed and justified.
        SmartMan: All facts need approval.
        """
        if approved_by != "smartman":
            # Submit for SmartMan approval
            from smartman.runtime.authority import smartman
            receipt = smartman.propose_action({
                "type": "add_fact",
                "fact": fact,
                "state_path": f"knowledge.facts.{fact.subject}_{fact.predicate}"
            })
            if receipt.approved_by != "smartman":
                raise PermissionError("Fact not approved")

        self.facts[self._fact_key(fact)] = fact

    def infer(self, query: str) -> List[Fact]:
        """
        Perform inference using rules + existing facts.
        Pattern from MeTTa: Composable reasoning chains.
        Returns: New facts with proof chains.
        """
        results = []

        # Parse query
        parsed = self._parse_query(query)

        # Apply inference rules
        for rule in self.rules:
            if rule.matches(parsed):
                new_facts = rule.apply(self.facts)
                results.extend(new_facts)

        return results

    def explain(self, fact: Fact) -> str:
        """
        Generate human-readable explanation of how we know this fact.
        Pattern from MeTTa: Proof tracking for transparency.
        """
        explanation = f"Fact: {fact.subject} {fact.predicate} {fact.object}\n"
        explanation += f"Confidence: {fact.confidence:.2%}\n"
        explanation += f"Source: {fact.source}\n"

        if fact.proof_chain:
            explanation += "\nProof chain:\n"
            for i, step in enumerate(fact.proof_chain, 1):
                explanation += f"  {i}. {step}\n"

        return explanation

@dataclass
class InferenceRule:
    """
    MeTTa-style inference rule.
    Pattern: If conditions met, derive new fact.
    """
    name: str
    conditions: List[str]  # Required facts
    conclusion: str        # Derived fact
    confidence_modifier: float = 1.0

    def matches(self, query) -> bool:
        """Check if rule applies to query"""
        pass

    def apply(self, facts: Dict[str, Fact]) -> List[Fact]:
        """Apply rule to generate new facts"""
        pass
```

### Example MeTTa Integration

```python
# Example: Using logic engine in Buddy

from smartman.runtime.brain.logic_engine import LogicEngine, Fact, InferenceRule

# Initialize logic engine
logic = LogicEngine()

# Add facts (with SmartMan approval)
logic.add_fact(Fact(
    subject="Paris",
    predicate="is_capital_of",
    object="France",
    confidence=1.0,
    source="wikidata",
    proof_chain=["Wikidata Q90 -> capital P36 -> Paris Q90"]
))

# Add inference rule
logic.rules.append(InferenceRule(
    name="capital_implies_city",
    conditions=["X is_capital_of Y"],
    conclusion="X is_city_in Y",
    confidence_modifier=1.0
))

# Perform inference
new_facts = logic.infer("what cities are in France?")
# Returns: Fact(subject="Paris", predicate="is_city_in", object="France", ...)

# Explain reasoning
explanation = logic.explain(new_facts[0])
print(explanation)
# Output:
# Fact: Paris is_city_in France
# Confidence: 100.00%
# Source: inferred
# Proof chain:
#   1. Paris is_capital_of France (from wikidata)
#   2. Applied rule: capital_implies_city
```

---

## Part 3: OpenClaw → SmartMan Session Management

### OpenClaw Donor Patterns

**What OpenClaw/Claude Code Provides**:
- **Session model**: Boot, resume, continuation
- **Tool routing**: Command → tool dispatch with permissions
- **Agent roles**: Planner, executor, validator separation
- **Memory logic**: Short-term session + long-term memory
- **Output contracts**: Structured results with receipts
- **Workflow ergonomics**: CLI flow, task continuation
- **Fallback behavior**: Offline mode, degraded mode

**SmartMan Landing**:
```python
# smartman/runtime/session_manager.py

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Session:
    """
    OpenClaw-inspired session model.
    Pattern: Resume-capable, state-aware sessions.
    """
    session_id: str
    started_at: str
    last_activity: str
    posture: str  # "witness", "operator", "authority"
    context: Dict[str, Any]
    task_queue: List[str]
    receipts: List[str]  # Receipt hashes from this session

class SessionManager:
    """
    OpenClaw-inspired session lifecycle management.
    Handles: boot, resume, continuation, cleanup.
    """

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.active_sessions: Dict[str, Session] = {}

    def start_session(self, posture: str = "operator") -> Session:
        """
        Pattern from OpenClaw: Boot sequence with clean state.
        """
        session_id = self._generate_session_id()
        session = Session(
            session_id=session_id,
            started_at=datetime.utcnow().isoformat() + "Z",
            last_activity=datetime.utcnow().isoformat() + "Z",
            posture=posture,
            context={},
            task_queue=[],
            receipts=[]
        )

        # Save session checkpoint
        self._save_session(session)
        self.active_sessions[session_id] = session

        return session

    def resume_session(self, session_id: str) -> Session:
        """
        Pattern from OpenClaw: Resume from checkpoint.
        SmartMan: All state preserved in receipts.
        """
        session_path = self.sessions_dir / f"{session_id}.json"

        if not session_path.exists():
            raise ValueError(f"Session {session_id} not found")

        # Load session state
        session_data = json.loads(session_path.read_text())
        session = Session(**session_data)

        # Update last activity
        session.last_activity = datetime.utcnow().isoformat() + "Z"

        self.active_sessions[session_id] = session
        return session

    def checkpoint_session(self, session_id: str):
        """
        Pattern from OpenClaw: Periodic checkpointing.
        LARQL: Like --resume flag, saves progress.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not active")

        session.last_activity = datetime.utcnow().isoformat() + "Z"
        self._save_session(session)

    def end_session(self, session_id: str, archive: bool = True):
        """
        Pattern from OpenClaw: Clean session closure.
        SmartMan: Archive receipts for audit.
        """
        session = self.active_sessions.pop(session_id, None)
        if not session:
            return

        if archive:
            # Archive session to permanent storage
            archive_path = self.sessions_dir / "archive" / f"{session_id}.json"
            archive_path.parent.mkdir(exist_ok=True, parents=True)
            archive_path.write_text(json.dumps(session.__dict__, indent=2))

        # Cleanup temp session file
        session_path = self.sessions_dir / f"{session_id}.json"
        if session_path.exists():
            session_path.unlink()
```

### Tool Router Integration (OpenClaw Pattern)

```python
# smartman/runtime/tool_router.py

from typing import Dict, Callable, Any
from dataclasses import dataclass

@dataclass
class Tool:
    """
    OpenClaw-inspired tool definition.
    Pattern: Each tool has permissions, capabilities, fallbacks.
    """
    name: str
    description: str
    handler: Callable
    requires_approval: bool = False
    offline_capable: bool = False
    fallback: Optional[str] = None

class ToolRouter:
    """
    OpenClaw-inspired tool routing with SmartMan permissions.
    Pattern: Command → permission check → tool execution → receipt.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_core_tools()

    def execute(self, tool_name: str, session: Session, **kwargs) -> Dict:
        """
        Pattern from OpenClaw: Tool routing with permission gates.
        SmartMan: All tool executions generate receipts.
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}

        # Check if approval needed
        if tool.requires_approval:
            from smartman.runtime.authority import smartman
            receipt = smartman.propose_action({
                "type": f"tool_{tool_name}",
                "session_id": session.session_id,
                "params": kwargs
            })

            if receipt.approved_by != "smartman":
                return {
                    "refused": True,
                    "reason": "SmartMan did not approve",
                    "receipt_hash": receipt.hash
                }

        # Execute tool
        try:
            result = tool.handler(**kwargs)

            # Record in session
            session.receipts.append(self._create_receipt_hash(tool_name, result))

            return {
                "success": True,
                "result": result,
                "tool": tool_name
            }

        except Exception as e:
            # Try fallback if available
            if tool.fallback:
                return self.execute(tool.fallback, session, **kwargs)

            return {
                "error": str(e),
                "tool": tool_name
            }
```

---

## Part 4: Complete Integration - Buddy Runtime

### Main Buddy Runtime (All Patterns Together)

```python
# smartman/runtime/buddy_companion.py

from pathlib import Path
from smartman.runtime.state.compact_state import CompactState
from smartman.runtime.brain.logic_engine import LogicEngine
from smartman.runtime.session_manager import SessionManager, Session
from smartman.runtime.tool_router import ToolRouter
from smartman.runtime.authority import SmartManAuthority

class BuddyCompanion:
    """
    Main Buddy runtime integrating all donor patterns.

    Architecture:
    - LARQL → Compact state (immutable base + receipts)
    - MeTTa → Logic engine (reasoning + proof tracking)
    - OpenClaw → Session management + tool routing
    - SmartMan → Authority over all mutations
    """

    def __init__(self, config_path: Path):
        # LARQL pattern: Immutable state
        self.state = CompactState(config_path)

        # MeTTa pattern: Logic engine
        self.logic = LogicEngine()

        # OpenClaw pattern: Session management
        self.sessions = SessionManager(config_path / "sessions")

        # OpenClaw pattern: Tool routing
        self.tools = ToolRouter()

        # SmartMan: Authority over all
        self.authority = SmartManAuthority(self.state)

        # Current session
        self.current_session: Optional[Session] = None

    def start(self, posture: str = "operator") -> Session:
        """
        Start new Buddy session.
        Pattern from OpenClaw: Clean boot sequence.
        """
        session = self.sessions.start_session(posture=posture)
        self.current_session = session

        print(f"✓ Buddy session started: {session.session_id}")
        print(f"  Posture: {session.posture}")
        print(f"  State: {len(self.state._receipts)} receipts loaded")

        return session

    def resume(self, session_id: str) -> Session:
        """
        Resume previous session.
        Pattern from OpenClaw: Resume from checkpoint.
        """
        session = self.sessions.resume_session(session_id)
        self.current_session = session

        print(f"✓ Buddy session resumed: {session_id}")
        print(f"  Last activity: {session.last_activity}")
        print(f"  Pending tasks: {len(session.task_queue)}")

        return session

    def process_command(self, command: str) -> Dict:
        """
        Process user command through full pipeline.

        Flow:
        1. Parse command
        2. Check posture permissions
        3. Route to appropriate tool
        4. Get SmartMan approval if needed
        5. Execute and record receipt
        6. Return result + explanation
        """
        if not self.current_session:
            return {"error": "No active session. Call start() or resume() first."}

        # Parse command (could use MeTTa logic engine here)
        parsed = self._parse_command(command)

        # Route to tool
        result = self.tools.execute(
            tool_name=parsed["tool"],
            session=self.current_session,
            **parsed["params"]
        )

        # Checkpoint session
        self.sessions.checkpoint_session(self.current_session.session_id)

        return result

    def ask(self, question: str) -> str:
        """
        Ask Buddy a question (uses logic engine + LLM).

        Pattern from MeTTa: Try reasoning first, LLM if needed.
        """
        # Try logic engine first
        inferred_facts = self.logic.infer(question)

        if inferred_facts:
            # We can answer from knowledge base
            fact = inferred_facts[0]
            explanation = self.logic.explain(fact)
            return explanation

        # Fall back to LLM
        from smartman.runtime.models.mind_router import mind
        response = mind.think(question)

        return response["response"]

    def shutdown(self, archive: bool = True):
        """
        Graceful shutdown with state preservation.
        Pattern from OpenClaw: Clean session closure.
        """
        if self.current_session:
            self.sessions.end_session(
                self.current_session.session_id,
                archive=archive
            )

        # Final state compact (like LARQL COMPILE)
        compact_path = self.state.base_path.parent / f"state_final_{datetime.utcnow().strftime('%Y%m%d')}.json"
        self.state.compact_base(compact_path)

        print("✓ Buddy shutdown complete")
        print(f"  State compacted to: {compact_path}")
```

---

## Part 5: SmartMan Authority (The Judge)

```python
# smartman/runtime/authority.py

from smartman.runtime.state.compact_state import CompactState, Receipt
from typing import Dict

class SmartManAuthority:
    """
    The immutable judge / gate system.
    Pattern: All mutations require approval.

    Integration:
    - LARQL: Immutable base enforcement
    - MeTTa: Logic-based approval rules
    - OpenClaw: Permission gates on tools
    """

    def __init__(self, state: CompactState):
        self.state = state
        self.rules = self._load_rules()

    def propose_action(self, action: Dict) -> Receipt:
        """
        Buddy proposes action, SmartMan approves or refuses.

        Approval logic:
        1. Check action against rules
        2. Validate state transition
        3. Check resource limits
        4. Generate receipt (approved or refused)
        """
        receipt = self.state.propose_change(
            path=action.get("state_path", "actions.log"),
            value=action
        )

        # Apply rules (could use MeTTa logic engine)
        if self._should_approve(action):
            receipt.approved_by = "smartman"
            receipt.source = "smartman_authority"
        else:
            receipt.approved_by = "refused"
            receipt.source = f"smartman_refused: {self._get_refusal_reason(action)}"

        return receipt

    def _should_approve(self, action: Dict) -> bool:
        """
        Approval decision logic.
        Pattern from MeTTa: Rule-based reasoning.
        """
        action_type = action.get("type")

        # Check against SmartMan rules
        for rule in self.rules:
            if rule.applies_to(action_type):
                if not rule.check(action):
                    return False

        return True

# Global SmartMan instance
smartman = None

def init_smartman(state: CompactState):
    global smartman
    smartman = SmartManAuthority(state)
```

---

## Part 6: Bringing It All Together

### Complete Startup Sequence

```python
# buddy_main.py - Entry point bringing all patterns together

from pathlib import Path
from smartman.runtime.buddy_companion import BuddyCompanion
from smartman.runtime.authority import init_smartman

def main():
    print("=" * 60)
    print("BUDDY AI - Integration Complete")
    print("Patterns: LARQL + MeTTa + OpenClaw → SmartMan → Buddy")
    print("=" * 60)

    # Initialize Buddy (loads all patterns)
    config_path = Path.home() / ".smartman"
    buddy = BuddyCompanion(config_path)

    # Initialize SmartMan authority
    init_smartman(buddy.state)

    # Start session (OpenClaw pattern)
    session = buddy.start(posture="operator")

    # Example: Ask question (MeTTa reasoning)
    answer = buddy.ask("What is the capital of France?")
    print(f"\nQ: What is the capital of France?")
    print(f"A: {answer}")

    # Example: Execute tool (OpenClaw routing + SmartMan approval)
    result = buddy.process_command("search github for recent PRs")
    print(f"\nTool execution: {result}")

    # Example: Add fact (MeTTa logic engine)
    from smartman.runtime.brain.logic_engine import Fact
    buddy.logic.add_fact(Fact(
        subject="London",
        predicate="is_capital_of",
        object="UK",
        confidence=1.0,
        source="user_input",
        proof_chain=["User stated this fact"]
    ))

    # Shutdown (LARQL compact + OpenClaw archive)
    buddy.shutdown(archive=True)

    print("\n✓ All patterns integrated successfully!")

if __name__ == "__main__":
    main()
```

---

## Summary: Complete Integration Map

### Pattern Sources → SmartMan Destinations

| Donor System | Pattern | SmartMan File | Purpose |
|--------------|---------|---------------|---------|
| **LARQL** | Immutable base + patches | `runtime/state/compact_state.py` | State management |
| **LARQL** | Walk-only mode | `runtime/models/model_router.py` | Resource awareness |
| **LARQL** | Template cache | `runtime/brain/workflow_cache.py` | Performance |
| **LARQL** | Profiling | `runtime/profiler.py` | Bottleneck detection |
| **LARQL** | Resumable ops | `runtime/session_manager.py` | Interrupt tolerance |
| **MeTTa** | Logic engine | `runtime/brain/logic_engine.py` | Reasoning |
| **MeTTa** | Knowledge graph | `runtime/brain/logic_engine.py` | Structured knowledge |
| **MeTTa** | Proof tracking | `runtime/brain/logic_engine.py` | Transparency |
| **MeTTa** | Type system | `runtime/brain/logic_engine.py` | Fact validation |
| **OpenClaw** | Session model | `runtime/session_manager.py` | Lifecycle management |
| **OpenClaw** | Tool routing | `runtime/tool_router.py` | Command dispatch |
| **OpenClaw** | Agent roles | `runtime/brain/planner.py` | Separation of concerns |
| **OpenClaw** | Memory logic | `runtime/session_manager.py` | Context preservation |
| **OpenClaw** | Output contracts | `runtime/receipt_engine.py` | Structured results |
| **OpenClaw** | Fallback behavior | `runtime/models/fallback_matrix.py` | Degraded modes |

### The Complete Flow

```
User Command
    ↓
Buddy (parse, route)
    ↓
SmartMan (approve/refuse) ← Uses MeTTa logic for decisions
    ↓
Tool Execution ← OpenClaw routing pattern
    ↓
Receipt Generation ← LARQL immutable overlay
    ↓
State Update ← LARQL compact state
    ↓
Session Checkpoint ← OpenClaw resume capability
    ↓
Return Result
```

---

## Next Steps: Implementation Order

1. **Week 1-2**: Implement `CompactState` (LARQL pattern)
2. **Week 3-4**: Implement `SessionManager` (OpenClaw pattern)
3. **Week 5-6**: Implement `LogicEngine` (MeTTa pattern)
4. **Week 7-8**: Implement `ToolRouter` (OpenClaw pattern)
5. **Week 9-10**: Integrate all into `BuddyCompanion`
6. **Week 11-12**: Wire `SmartManAuthority` as gate
7. **Week 13-16**: Test, profile, optimize
8. **Week 17-20**: Source 2 visualization (shell layer)

---

## The Vision Realized

**Three donor systems**, **one authority**, **one operator**, **infinite possibilities**.

LARQL teaches us local knowledge management.
MeTTa teaches us logical reasoning.
OpenClaw teaches us session ergonomics.

SmartMan enforces the law.
Buddy operates the tools.
Source 2 visualizes the state.

**This is BUDDY CLAW - complete integration.**

All extensions linked. All patterns harvested. All flowing through SmartMan into Buddy.

---

**End of integration guide.**
