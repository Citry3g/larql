# Buddy AI - Source 2 Integration Master Design

**Vision**: An AI companion living inside Source 2/s&box with mind (LLM), arm (tools), and Steam integration - all expertly designed for game development workflows.

**Philosophy**: Your AI, your data, your rules. Local-first, privacy-preserving, game-integrated.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SOURCE 2 / s&box                             │
│                  (3D Visualization Shell)                        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Workspace   │  │   Tools      │  │   Receipt    │         │
│  │  Room        │  │   Palette    │  │   Wall       │         │
│  │              │  │              │  │              │         │
│  │  [Tasks as   │  │  [Browser]   │  │  [Audit      │         │
│  │   objects]   │  │  [Terminal]  │  │   trail]     │         │
│  │              │  │  [GDrive]    │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  Lua Interface Layer ↕ C++ Bridge ↕ Python Runtime             │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    BUDDY RUNTIME (Python)                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    MIND (LLM Layer)                       │  │
│  │                                                           │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ BitNet   │→ │  Cache   │→ │  GPT-4   │→ [Refuse]    │  │
│  │  │ Local    │  │  Memory  │  │  Remote  │              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  │       ↓ Fallback Cascade                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↕                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  SMARTMAN (Authority)                     │  │
│  │                                                           │  │
│  │  All proposals → Approve/Refuse → Receipt generation     │  │
│  │  Immutable base + Receipt overlay = Current state        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↕                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    ARM (Tools)                            │  │
│  │                                                           │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │  │
│  │  │ GitHub   │  │ GDrive   │  │ Browser  │  +10 more    │  │
│  │  │ API      │  │ API      │  │ Selenium │              │  │
│  │  └──────────┘  └──────────┘  └──────────┘              │  │
│  │       ↓ Batch processor for efficiency                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  State: ~/.buddy/state.json (base) + receipts.jsonl (log)      │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    STEAM INTEGRATION                             │
│                                                                  │
│  • Workshop uploads (maps, mods, assets)                        │
│  • Cloud save sync (state backups)                              │
│  • Friend list integration (multiplayer Buddy sessions)         │
│  • Achievement tracking (development milestones)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: MIND (LLM System)

### Local BitNet Inference (Offline Brain)

**Implementation**:
```python
# runtime/models/bitnet_brain.py

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

class BitNetBrain:
    """
    Local 1.58-bit neural network for offline inference.
    Based on BitNet research: 55-82% energy savings, CPU-friendly.
    """

    def __init__(self, model_path: Path):
        """
        Load BitNet weights from mmap'd file.
        Pattern from LARQL: mmap for large weights, zero-copy.
        """
        self.model_path = model_path
        self.weights = self._mmap_weights()
        self.cache = {}  # Response cache

    def _mmap_weights(self) -> np.ndarray:
        """Memory-map BitNet weights (1.58-bit quantized)"""
        import mmap
        # Implementation: Load .bitnet file format
        # Weights are ternary: -1, 0, +1 (1.58 bits per parameter)
        pass

    def infer(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Run inference on BitNet model.
        Fast, local, no API calls.
        """
        # Check cache first
        cache_key = f"{prompt}:{max_tokens}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Tokenize
        tokens = self.tokenize(prompt)

        # Run forward pass (CPU-optimized)
        # BitNet uses INT8 activations + 1.58-bit weights
        output_tokens = self._forward_pass(tokens, max_tokens)

        # Decode
        result = self.detokenize(output_tokens)

        # Cache result
        self.cache[cache_key] = result

        return result

    def _forward_pass(self, input_ids: List[int], max_tokens: int) -> List[int]:
        """
        BitNet forward pass optimized for CPU.
        Uses ternary weights: multiply becomes addition/subtraction.
        """
        # Implementation based on BitNet paper
        # Key insight: -1/0/+1 weights → no float multiply, just add/sub
        pass
```

### Fallback Cascade (Mind Router)

**Implementation**:
```python
# runtime/models/mind_router.py

from runtime.models.fallback_router import FallbackCascade
from runtime.models.bitnet_brain import BitNetBrain
from runtime.profiler import profiler

class MindRouter:
    """
    Routes inference requests through fallback chain.
    Pattern from LARQL: Try fast local first, expensive remote last.
    """

    def __init__(self):
        self.cascade = FallbackCascade(name="mind")
        self._setup_cascade()

    def _setup_cascade(self):
        """Setup inference fallback chain"""

        # Method 1: BitNet local (fastest, offline)
        self.bitnet = BitNetBrain(Path("~/.buddy/models/bitnet-1.58b.model"))
        self.cascade.add_method(
            name="bitnet_local",
            handler=self._try_bitnet,
            availability_check=lambda: self.bitnet.is_loaded()
        )

        # Method 2: Response cache (instant)
        self.cascade.add_method(
            name="cache_hit",
            handler=self._try_cache,
            availability_check=lambda: True
        )

        # Method 3: GPT-4 API (slow, requires internet)
        self.cascade.add_method(
            name="gpt4_remote",
            handler=self._try_gpt4,
            availability_check=self._is_online
        )

        # Method 4: Refuse with helpful message
        self.cascade.add_method(
            name="refuse_gracefully",
            handler=self._refuse,
            availability_check=lambda: True
        )

    @profiler.measure("bitnet_inference")
    def _try_bitnet(self, request: dict) -> str:
        """Try BitNet local inference"""
        return self.bitnet.infer(request["prompt"])

    @profiler.measure("gpt4_api_call")
    def _try_gpt4(self, request: dict) -> str:
        """Try GPT-4 API (when online)"""
        import openai
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": request["prompt"]}]
        )
        return response.choices[0].message.content

    def think(self, prompt: str) -> Dict:
        """
        Main inference entry point.
        Returns response + metadata about which method was used.
        """
        result = self.cascade.execute({"prompt": prompt})

        return {
            "response": result.result,
            "method": result.method_used,
            "fallbacks_tried": result.fallbacks_tried,
            "offline_capable": result.method_used == "bitnet_local"
        }
```

---

## Component 2: SMARTMAN (Authority Layer)

**Implementation**:
```python
# runtime/smartman_authority.py

from runtime.state.immutable_state import ImmutableStateManager, Receipt
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SmartManAuthority:
    """
    The law enforcement layer.
    All state changes must be approved here.
    Pattern from LARQL: Immutable base + approved overlays only.
    """

    def __init__(self, state_path: Path):
        self.state = ImmutableStateManager(
            base_path=state_path / "state.json",
            receipts_path=state_path / "receipts.jsonl"
        )

        # Load SmartMan rules
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict:
        """Load SmartMan governance rules"""
        return {
            # File system rules
            "allowed_write_paths": [
                "~/Documents/buddy_workspace/**",
                "~/.buddy/temp/**"
            ],
            "forbidden_paths": [
                "/System/**",
                "~/.ssh/**",
                "~/.buddy/state.json"  # Only SmartMan can modify base
            ],

            # Tool rules
            "tools_require_approval": [
                "github_push",
                "gdrive_delete",
                "terminal_sudo"
            ],

            # Resource limits
            "max_api_calls_per_hour": 100,
            "max_file_size_mb": 50,
            "max_session_memory_mb": 2000
        }

    def propose_action(self, action: Dict) -> Receipt:
        """
        Buddy proposes an action.
        SmartMan generates a receipt (approved or refused).
        """
        # Create receipt proposal
        receipt = self.state.propose_change(
            path=action["state_path"],
            value=action["value"],
            operation=action.get("operation", "set"),
            approved_by="pending"
        )

        # Check against rules
        if self._approve(action, receipt):
            receipt.approved_by = "smartman"
            logger.info(f"✓ SmartMan APPROVED: {action['type']}")
        else:
            receipt.approved_by = "refused"
            logger.warning(f"✗ SmartMan REFUSED: {action['type']}")

        return receipt

    def _approve(self, action: Dict, receipt: Receipt) -> bool:
        """
        Decision logic for approval/refusal.
        Returns True if approved, False if refused.
        """
        action_type = action.get("type")

        # Check tool whitelist
        if action_type in self.rules["tools_require_approval"]:
            # Require explicit user confirmation for sensitive tools
            return self._get_user_confirmation(action)

        # Check file paths
        if action_type == "file_write":
            target_path = action.get("path")
            if self._is_forbidden_path(target_path):
                return False

        # Check resource limits
        if action_type == "api_call":
            if self._exceeds_rate_limit():
                return False

        # Default: approve
        return True

    def apply_receipt(self, receipt: Receipt) -> bool:
        """
        Apply approved receipt to state.
        Only SmartMan can call this.
        """
        if receipt.approved_by != "smartman":
            raise PermissionError("Only approved receipts can be applied")

        self.state.apply_receipt(receipt)
        return True

    def get_audit_trail(self) -> List[Receipt]:
        """
        Get complete audit trail of all state changes.
        For Source 2 receipt wall visualization.
        """
        return self.state._receipts
```

---

## Component 3: ARM (Tool Integration)

**Implementation**:
```python
# runtime/tools/tool_registry.py

from typing import Dict, Callable, List
from dataclasses import dataclass
from runtime.batch_processor import BatchProcessor, BatchConfig

@dataclass
class Tool:
    name: str
    description: str
    handler: Callable
    requires_approval: bool = False
    batch_capable: bool = False

class ToolRegistry:
    """
    Central registry for all Buddy tools.
    Pattern from LARQL: Batch similar operations for efficiency.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_core_tools()

    def _register_core_tools(self):
        """Register all available tools"""

        # GitHub integration
        self.register(Tool(
            name="github_list_prs",
            description="List pull requests from repository",
            handler=self._github_list_prs,
            batch_capable=True
        ))

        self.register(Tool(
            name="github_create_pr",
            description="Create new pull request",
            handler=self._github_create_pr,
            requires_approval=True
        ))

        # Google Drive integration
        self.register(Tool(
            name="gdrive_search",
            description="Search files in Google Drive",
            handler=self._gdrive_search,
            batch_capable=True
        ))

        # Browser automation
        self.register(Tool(
            name="browser_navigate",
            description="Navigate to URL and extract content",
            handler=self._browser_navigate
        ))

        # File system
        self.register(Tool(
            name="file_read",
            description="Read file contents",
            handler=self._file_read,
            batch_capable=True
        ))

        self.register(Tool(
            name="file_write",
            description="Write file contents",
            handler=self._file_write,
            requires_approval=True
        ))

    def register(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool

    def execute(self, tool_name: str, **kwargs) -> Dict:
        """Execute a tool (with SmartMan approval if needed)"""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Check if approval needed
        if tool.requires_approval:
            # Generate proposal for SmartMan
            from runtime.smartman_authority import smartman
            receipt = smartman.propose_action({
                "type": f"tool_{tool_name}",
                "state_path": f"tools.{tool_name}.last_call",
                "value": kwargs
            })

            if receipt.approved_by != "smartman":
                return {
                    "success": False,
                    "error": "Action refused by SmartMan",
                    "receipt": receipt.hash
                }

        # Execute tool
        result = tool.handler(**kwargs)
        return {"success": True, "result": result}

    # Tool implementations
    def _github_list_prs(self, repo: str) -> List[Dict]:
        """List GitHub PRs"""
        import requests
        response = requests.get(f"https://api.github.com/repos/{repo}/pulls")
        return response.json()

    def _file_read(self, path: str) -> str:
        """Read file"""
        from pathlib import Path
        return Path(path).read_text()

    def _file_write(self, path: str, content: str) -> bool:
        """Write file (requires SmartMan approval)"""
        from pathlib import Path
        Path(path).write_text(content)
        return True
```

---

## Component 4: SOURCE 2 Integration

### Lua → Python Bridge

**Implementation** (Lua side in s&box):
```lua
-- addons/buddy_ai/lua/autorun/buddy_bridge.lua

BuddyBridge = BuddyBridge or {}

-- HTTP bridge to Python runtime
BuddyBridge.RuntimeURL = "http://localhost:8765/buddy"

-- Send request to Buddy runtime
function BuddyBridge:SendRequest(action, data)
    HTTP({
        method = "POST",
        url = self.RuntimeURL .. "/action",
        parameters = {
            action = action,
            data = util.TableToJSON(data)
        },
        success = function(code, body)
            local response = util.JSONToTable(body)
            BuddyBridge:HandleResponse(response)
        end,
        failed = function(error)
            print("[Buddy] Connection failed:", error)
        end
    })
end

-- Handle response from Buddy
function BuddyBridge:HandleResponse(response)
    if response.type == "task_update" then
        BuddyUI:UpdateTaskDisplay(response.data)
    elseif response.type == "receipt" then
        BuddyUI:AddReceipt(response.data)
    elseif response.type == "message" then
        BuddyUI:ShowMessage(response.data.text)
    end
end

-- Ask Buddy to perform action
function BuddyBridge:Ask(prompt)
    self:SendRequest("query", {prompt = prompt})
end

-- Get current state
function BuddyBridge:GetState()
    self:SendRequest("get_state", {})
end
```

### Python → Lua Bridge

**Implementation** (Python side):
```python
# runtime/source2_bridge.py

from flask import Flask, request, jsonify
from typing import Dict
import threading

class Source2Bridge:
    """
    HTTP bridge between Buddy runtime and Source 2.
    Pattern: Bidirectional communication via JSON packets.
    """

    def __init__(self, port: int = 8765):
        self.app = Flask(__name__)
        self.port = port
        self._setup_routes()
        self.callbacks = {}

    def _setup_routes(self):
        """Setup HTTP endpoints"""

        @self.app.route('/buddy/action', methods=['POST'])
        def handle_action():
            """Receive action from Source 2"""
            data = request.json
            action = data.get('action')
            params = data.get('data', {})

            # Route to appropriate handler
            if action == "query":
                result = self._handle_query(params)
            elif action == "get_state":
                result = self._handle_get_state(params)
            else:
                result = {"error": f"Unknown action: {action}"}

            return jsonify(result)

    def _handle_query(self, params: Dict) -> Dict:
        """Handle query from Source 2"""
        from runtime.buddy_companion import buddy

        prompt = params.get('prompt')
        response = buddy.think(prompt)

        return {
            "type": "message",
            "data": {
                "text": response["response"],
                "method": response["method"]
            }
        }

    def _handle_get_state(self, params: Dict) -> Dict:
        """Get current Buddy state"""
        from runtime.smartman_authority import smartman

        state = smartman.state.current_state()
        return {
            "type": "state",
            "data": state
        }

    def start(self):
        """Start bridge server in background thread"""
        thread = threading.Thread(
            target=lambda: self.app.run(port=self.port, threaded=True)
        )
        thread.daemon = True
        thread.start()
        print(f"✓ Source 2 bridge running on port {self.port}")
```

### 3D Workspace Visualization

**Source 2 Map Layout**:
```
// maps/buddy_workspace.vmap

entities:
- workspace_room: Main 3D space with floating task cards
- tool_palette: Wall of tool icons (browser, terminal, GitHub, etc.)
- receipt_wall: Scrollable audit trail of all actions
- buddy_avatar: 3D character representing Buddy AI
- smartman_gate: Visual representation of approval/refusal

Task Card (3D Model):
- Title: Floating holographic text
- Status: Color-coded (pending=yellow, approved=green, refused=red)
- Progress: Visual progress bar
- Interactions: Click to expand, drag to reorder
```

**Lua Visualization Code**:
```lua
-- addons/buddy_ai/lua/ui/workspace.lua

BuddyWorkspace = {}

-- Create task card entity
function BuddyWorkspace:CreateTaskCard(task_data)
    local card = ents.Create("buddy_task_card")
    card:SetPos(self:GetNextCardPosition())
    card:SetTaskData(task_data)

    -- Color by status
    if task_data.status == "approved" then
        card:SetColor(Color(0, 255, 0, 200))  -- Green
    elseif task_data.status == "refused" then
        card:SetColor(Color(255, 0, 0, 200))  -- Red
    else
        card:SetColor(Color(255, 255, 0, 200))  -- Yellow
    end

    card:Spawn()
    return card
end

-- Update receipt wall
function BuddyWorkspace:AddReceipt(receipt_data)
    local receipt = ents.Create("buddy_receipt")
    receipt:SetPos(self.ReceiptWall:GetNextSlot())
    receipt:SetReceiptData(receipt_data)
    receipt:Spawn()

    -- Auto-scroll wall
    self.ReceiptWall:ScrollToLatest()
end
```

---

## Component 5: STEAM Integration

**Implementation**:
```python
# runtime/steam_integration.py

import os
from pathlib import Path
from typing import Dict, List

class SteamIntegration:
    """
    Integration with Steam Workshop and Cloud.
    For sharing Buddy configurations and backing up state.
    """

    def __init__(self, app_id: str = "590720"):  # s&box app ID
        self.app_id = app_id
        self.workshop_path = self._get_workshop_path()

    def _get_workshop_path(self) -> Path:
        """Get Steam Workshop content path"""
        # Platform-specific paths
        if os.name == 'nt':  # Windows
            base = Path(os.environ['PROGRAMFILES(X86)']) / "Steam"
        else:  # Linux/Mac
            base = Path.home() / ".local/share/Steam"

        return base / "steamapps/workshop/content" / self.app_id

    def upload_buddy_config(self, config_name: str, description: str) -> str:
        """
        Upload Buddy configuration to Steam Workshop.
        Includes: Buddyfile, custom tools, receipt templates.
        """
        from steam.webapi import WebAPI

        # Package configuration
        package_path = self._package_config(config_name)

        # Upload to Workshop
        workshop_id = self._upload_to_workshop(
            package_path,
            title=f"Buddy Config: {config_name}",
            description=description
        )

        return workshop_id

    def sync_state_to_cloud(self) -> bool:
        """
        Backup Buddy state to Steam Cloud.
        Includes: state.json, receipts.jsonl, session history.
        """
        from steam.client import SteamClient

        # Get state files
        state_files = [
            "~/.buddy/state.json",
            "~/.buddy/receipts.jsonl",
            "~/.buddy/session_history.json"
        ]

        # Upload to Steam Cloud
        for file_path in state_files:
            self._upload_to_cloud(Path(file_path))

        return True

    def download_buddy_config(self, workshop_id: str) -> Path:
        """
        Download Buddy configuration from Workshop.
        Auto-installs tools and applies settings.
        """
        # Download from Workshop
        config_path = self._download_from_workshop(workshop_id)

        # Apply configuration
        self._apply_config(config_path)

        return config_path
```

---

## Component 6: Game Designer Workflow

### Buddy as Game Dev Assistant

**Use Cases**:

1. **Asset Management**
   - "Find all unused textures in the project"
   - "Optimize these models for mobile"
   - "Generate LODs for terrain meshes"

2. **Code Generation**
   - "Create a weapon system with recoil patterns"
   - "Implement A* pathfinding for NPCs"
   - "Add multiplayer sync for this component"

3. **Level Design**
   - "Generate procedural dungeon layout"
   - "Balance lighting in this scene"
   - "Place cover points strategically"

4. **Testing & QA**
   - "Run playtests and collect metrics"
   - "Analyze framerate bottlenecks"
   - "Generate test cases for this mechanic"

**Implementation**:
```python
# runtime/tools/gamedev_tools.py

class GameDevTools:
    """Game development-specific tools for Source 2"""

    def analyze_scene(self, scene_path: str) -> Dict:
        """Analyze Source 2 scene for optimization opportunities"""
        # Parse .vmap file
        # Count draw calls, check texture sizes, analyze hierarchy
        pass

    def generate_code(self, spec: str, language: str = "lua") -> str:
        """Generate game code from natural language spec"""
        from runtime.models.mind_router import mind

        prompt = f"""
        Generate {language} code for Source 2/s&box:
        Specification: {spec}

        Requirements:
        - Follow s&box best practices
        - Include comments
        - Add error handling
        - Use appropriate entity system
        """

        response = mind.think(prompt)
        return response["response"]

    def optimize_assets(self, asset_list: List[str]) -> Dict:
        """Batch optimize game assets"""
        from runtime.batch_processor import BatchProcessor

        def optimize_single(asset_path: str) -> Dict:
            # Run optimization (compress textures, reduce polys, etc.)
            pass

        processor = BatchProcessor(
            batch_handler=lambda batch: [optimize_single(a) for a in batch],
            config=BatchConfig(batch_size=10)
        )

        with processor:
            for asset in asset_list:
                processor.add(asset)

        return {"optimized": len(asset_list)}
```

---

## Complete Integration Example

**Buddyfile.yml** (Declarative configuration):
```yaml
# Buddy configuration for game development

base: "smartman-approved-state-v1.0"

mind:
  primary: "bitnet-local"
  fallback:
    - "cache"
    - "gpt4-remote"
  cache_size_mb: 500

tools:
  enabled:
    - github
    - gdrive
    - browser
    - terminal
    - gamedev  # Game-specific tools

  gamedev:
    source2_sdk_path: "/mnt/c/s&box"
    auto_optimize_assets: true
    generate_lods: true

smartman:
  approval_required:
    - "github_push"
    - "file_delete"
    - "terminal_sudo"

  allowed_paths:
    - "~/Documents/gamedev/**"
    - "~/.buddy/workspace/**"

source2:
  bridge_port: 8765
  workspace_map: "maps/buddy_workspace.vmap"
  auto_launch: true

steam:
  sync_to_cloud: true
  workshop_auto_update: false
  app_id: "590720"  # s&box

receipts:
  visualize_in_3d: true
  max_history: 10000
  compact_interval: "weekly"

posture: "operator"  # witness | operator | authority
```

**Startup Sequence**:
```python
# buddy_main.py - Entry point

from pathlib import Path
from runtime.buddy_companion import BuddyCompanion
from runtime.smartman_authority import SmartManAuthority
from runtime.source2_bridge import Source2Bridge
from runtime.steam_integration import SteamIntegration
from runtime.profiler import profiler

def main():
    print("=== Buddy AI - Source 2 Edition ===")

    # 1. Initialize SmartMan (authority layer)
    print("Initializing SmartMan authority...")
    smartman = SmartManAuthority(Path("~/.buddy"))

    # 2. Initialize Buddy (operator layer)
    print("Initializing Buddy companion...")
    buddy = BuddyCompanion(smartman)

    # 3. Start Source 2 bridge
    print("Starting Source 2 bridge...")
    bridge = Source2Bridge(port=8765)
    bridge.start()

    # 4. Initialize Steam integration
    print("Connecting to Steam...")
    steam = SteamIntegration()
    steam.sync_state_to_cloud()

    # 5. Load BitNet brain
    print("Loading BitNet local brain...")
    buddy.mind.load_bitnet()

    # 6. Ready
    print("\n✓ Buddy is ready!")
    print("  - Mind: BitNet local (offline capable)")
    print("  - Arm: 15 tools loaded")
    print("  - Authority: SmartMan active")
    print("  - Source 2: Bridge running on :8765")
    print("  - Steam: Cloud sync enabled")
    print("\nOpen Source 2 and load 'buddy_workspace' map.")

    # 7. Main loop (keep alive)
    try:
        while True:
            # Periodic tasks
            if profiler._enabled and buddy.request_count % 100 == 0:
                print(profiler.report())

            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down Buddy...")
        steam.sync_state_to_cloud()  # Final backup
        print("✓ State synced to Steam Cloud")
        print("Goodbye!")

if __name__ == "__main__":
    main()
```

---

## Development Roadmap

### Phase 1: Foundation (Weeks 1-4)
- [ ] Implement ImmutableStateManager
- [ ] Implement ProgressiveProfiler
- [ ] Implement SmartManAuthority
- [ ] Basic tool registry (file ops, GitHub)
- [ ] Python HTTP bridge

### Phase 2: Mind (Weeks 5-8)
- [ ] BitNet model integration
- [ ] Fallback cascade router
- [ ] Response caching
- [ ] Profiling of inference bottlenecks

### Phase 3: Source 2 (Weeks 9-12)
- [ ] Lua bridge implementation
- [ ] 3D workspace map design
- [ ] Task card entities
- [ ] Receipt wall visualization
- [ ] Buddy avatar

### Phase 4: Steam (Weeks 13-16)
- [ ] Workshop upload/download
- [ ] Cloud save sync
- [ ] Configuration sharing
- [ ] Achievement system

### Phase 5: Game Dev Tools (Weeks 17-20)
- [ ] Asset optimization tools
- [ ] Code generation
- [ ] Scene analysis
- [ ] Playtest automation

---

## Success Metrics

After 6 months:
- ✅ Buddy runs 100% offline with BitNet
- ✅ All actions logged via receipts (full audit trail)
- ✅ SmartMan enforces security (zero bypasses)
- ✅ Source 2 workspace fully functional
- ✅ Steam integration complete
- ✅ 50+ game dev tasks automated
- ✅ Sub-100ms response time for common queries
- ✅ <500MB memory footprint

---

## The Vision Realized

**What you'll have**:

A 3D workspace inside Source 2 where you walk around and interact with Buddy AI:
- Tasks appear as floating cards
- Tools are on the wall like weapons in a game
- Receipts scroll on an audit wall
- Buddy responds through a character avatar
- Everything syncs to Steam Cloud
- Works offline with BitNet
- SmartMan always enforces the rules

**Your AI, your data, your game development workflow - all in one beautiful 3D environment.**

This is **BUDDY in SOURCE 2** - expertly designed for game development. 🎮🤖

---

## License & Attribution

- **Architecture**: Adapted from LARQL patterns (Apache-2.0)
- **BitNet**: Based on Microsoft Research paper
- **Source 2**: Valve/s&box SDK
- **Your Code**: Your license, your rules

Start building. The blueprint is complete.
