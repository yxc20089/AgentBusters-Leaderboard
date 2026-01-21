#!/usr/bin/env python3
"""
A2A Evaluation Script

Triggers evaluation of a Purple Agent through the Green Agent A2A server.

Usage:
    python scripts/run_a2a_eval.py --green-url http://localhost:9109 --purple-url http://localhost:9110

Examples:
    # Quick test with 5 tasks
    python scripts/run_a2a_eval.py --num-tasks 5

    # Full evaluation with debate
    python scripts/run_a2a_eval.py --num-tasks 100 --conduct-debate

    # Custom URLs
    python scripts/run_a2a_eval.py --green-url http://10.0.0.1:9109 --purple-url http://10.0.0.2:9110
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import httpx


async def send_eval_request(
    green_url: str,
    purple_url: str,
    num_tasks: int,
    conduct_debate: bool,
    timeout: int,
    verbose: bool,
) -> dict:
    """Send evaluation request to Green Agent."""
    
    # Build the evaluation request
    eval_request = {
        "participants": {
            "purple_agent": purple_url
        },
        "config": {
            "num_tasks": num_tasks,
            "conduct_debate": conduct_debate,
        }
    }
    
    # Build A2A JSON-RPC message
    message_id = f"msg-{uuid4().hex[:8]}"
    request_id = f"eval-{uuid4().hex[:8]}"
    
    a2a_request = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": request_id,
        "params": {
            "message": {
                "messageId": message_id,
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": json.dumps(eval_request)
                    }
                ]
            }
        }
    }
    
    if verbose:
        print(f"\n📤 Sending evaluation request to {green_url}")
        print(f"   Purple Agent: {purple_url}")
        print(f"   Num Tasks: {num_tasks}")
        print(f"   Conduct Debate: {conduct_debate}")
        print(f"   Message ID: {message_id}")
        print()
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                green_url,
                json=a2a_request,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            return {"error": f"Timeout after {timeout}s - evaluation may take longer for large datasets"}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": str(e)}


def print_results(result: dict, output_file: str = None):
    """Print and optionally save results."""
    
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return False

    _inject_llm_outputs(result)

    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    if "result" in result:
        data = result["result"]
        
        # Check if it's a task response
        if isinstance(data, dict):
            task = data.get("task", data)
            
            # Print task status
            status = task.get("status", {})
            state = status.get("state", "unknown")
            print(f"\n🔹 Task State: {state}")
            
            # Print artifacts if available
            artifacts = task.get("artifacts", [])
            for i, artifact in enumerate(artifacts):
                print(f"\n📎 Artifact {i+1}: {artifact.get('name', 'Unnamed')}")
                parts = artifact.get("parts", [])
                for part in parts:
                    root = part.get("root", part)
                    if root.get("kind") == "text":
                        print(root.get("text", ""))
                    elif root.get("kind") == "data" or "data" in root:
                        print(json.dumps(root.get("data", {}), indent=2))
    else:
        print(json.dumps(result, indent=2))
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    return True


def _inject_llm_outputs(result: dict) -> None:
    """Attach aggregated LLM raw outputs to the evaluation data, if available."""
    if not isinstance(result, dict):
        return
    task = None
    if "result" in result and isinstance(result["result"], dict):
        task = result["result"].get("task", result["result"])
    if not isinstance(task, dict):
        return

    artifacts = task.get("artifacts", [])
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            root = part.get("root", part)
            if not isinstance(root, dict):
                continue
            data = None
            if root.get("kind") == "data":
                data = root.get("data")
            elif "data" in root:
                data = root.get("data")
            if not isinstance(data, dict):
                continue

            results = data.get("results")
            if not isinstance(results, list):
                continue

            llm_outputs: dict[str, dict[str, str]] = {}
            for idx, entry in enumerate(results):
                if not isinstance(entry, dict):
                    continue
                raw_output = entry.get("llm_raw_output")
                if not raw_output:
                    continue
                dataset = entry.get("dataset_type", "unknown")
                example_id = entry.get("example_id") or entry.get("task_id") or str(idx)
                llm_outputs.setdefault(dataset, {})[example_id] = raw_output

            data["llm_outputs"] = llm_outputs


async def poll_for_completion(
    green_url: str,
    task_id: str,
    context_id: str,
    timeout: int,
    poll_interval: int = 5,
):
    """Poll for task completion (for streaming/long-running evaluations)."""
    
    get_task_request = {
        "jsonrpc": "2.0",
        "method": "tasks/get",
        "id": f"get-{uuid4().hex[:8]}",
        "params": {
            "id": task_id
        }
    }
    
    start_time = datetime.now()
    async with httpx.AsyncClient(timeout=30) as client:
        while (datetime.now() - start_time).seconds < timeout:
            try:
                response = await client.post(green_url, json=get_task_request)
                result = response.json()
                
                if "result" in result:
                    task = result["result"]
                    state = task.get("status", {}).get("state", "")
                    
                    if state in ("completed", "failed", "rejected", "canceled"):
                        return result
                    
                    print(f"⏳ Task state: {state}... (elapsed: {(datetime.now() - start_time).seconds}s)")
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                print(f"⚠️ Poll error: {e}")
                await asyncio.sleep(poll_interval)
        
        return {"error": f"Timeout after {timeout}s waiting for completion"}


def main():
    parser = argparse.ArgumentParser(
        description="Run A2A evaluation of Purple Agent through Green Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--green-url",
        type=str,
        default="http://localhost:9109",
        help="Green Agent A2A server URL (default: http://localhost:9109)"
    )
    parser.add_argument(
        "--purple-url",
        type=str,
        default="http://localhost:9110",
        help="Purple Agent A2A server URL (default: http://localhost:9110)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=10,
        help="Number of evaluation tasks to run (default: 10)"
    )
    parser.add_argument(
        "--conduct-debate",
        action="store_true",
        help="Enable adversarial debate during evaluation"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("🚀 FAB++ A2A Evaluation")
    print(f"   Green Agent: {args.green_url}")
    print(f"   Purple Agent: {args.purple_url}")
    print(f"   Tasks: {args.num_tasks}")
    print(f"   Timeout: {args.timeout}s")
    
    # Run evaluation
    result = asyncio.run(send_eval_request(
        green_url=args.green_url,
        purple_url=args.purple_url,
        num_tasks=args.num_tasks,
        conduct_debate=args.conduct_debate,
        timeout=args.timeout,
        verbose=args.verbose,
    ))
    
    # Print results
    success = print_results(result, args.output)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
