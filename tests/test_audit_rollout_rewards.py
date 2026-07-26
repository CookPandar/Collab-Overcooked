import unittest

from scripts import audit_rollout_rewards as audit


class AuditRolloutRewardsTest(unittest.TestCase):
    def test_action_block_strips_fenced_action_language_label(self):
        text = (
            "```\n"
            "Think: ok\n"
            "Recent Goal: use the oven.\n"
            "Action:\n"
            "```Chef\n"
            "pickup(oven0)\n"
            "```\n"
        )

        action = audit.action_block(text)

        self.assertEqual(action, "pickup(oven0)")
        self.assertEqual(audit.classify_action(action, text), "embodied")

    def test_same_timestep_matching_request_can_appear_after_response_in_storage(self):
        rows = [
            {
                "file": "rollout_rank4_u00008.pt",
                "idx": 10,
                "agent": 0,
                "ts": 5,
                "mode": "embodied",
                "embodied": "pickup(bell_pepper, counter)",
                "action": "pickup(bell_pepper, counter)",
                "paired_comm": 0.5,
            },
            {
                "file": "rollout_rank4_u00008.pt",
                "idx": 15,
                "agent": 1,
                "ts": 5,
                "mode": "communication",
                "embodied": "",
                "action": "Collab(request(Chef, pickup(bell_pepper, counter)))",
                "paired_comm": 0.5,
            },
        ]

        issues = audit.find_paired_comm_mismatch_issues(rows)

        self.assertEqual(issues, [])

    def test_positive_response_with_bad_saved_meta_is_flagged(self):
        rows = [
            {
                "file": "rollout_rank0_u00001.pt",
                "idx": 0,
                "agent": 1,
                "ts": 2,
                "mode": "embodied",
                "embodied": "place_obj_on_counter()",
                "action": "place_obj_on_counter()",
                "paired_comm": 0.5,
                "paired_comm_role": "responder",
                "paired_comm_result": "helpful_request_accepted",
                "paired_comm_request_action": "pickup(onion,counter)",
            }
        ]

        issues = audit.find_paired_comm_mismatch_issues(rows)

        self.assertEqual(
            [kind for kind, _ in issues],
            ["PAIRED_COMM_POSITIVE_RESPONSE_BAD_META"],
        )

    def test_accepted_response_requires_rewarded_request_row(self):
        rows = [
            {
                "file": "rollout_rank2_u00011.pt",
                "idx": 0,
                "agent": 0,
                "ts": 3,
                "mode": "communication",
                "embodied": "",
                "action": "Collab(request(Assistant,place_obj_on_counter()))",
                "paired_comm": 0.0,
            },
            {
                "file": "rollout_rank2_u00011.pt",
                "idx": 8,
                "agent": 1,
                "ts": 3,
                "mode": "communication",
                "embodied": "",
                "action": "Collab(ack(Chef))",
                "paired_comm": 0.5,
                "paired_comm_role": "responder",
                "paired_comm_result": "helpful_request_accepted",
                "paired_comm_request_action": "place_obj_on_counter()",
            },
        ]

        issues = audit.find_paired_comm_mismatch_issues(rows)

        self.assertEqual(
            [kind for kind, _ in issues],
            ["PAIRED_COMM_ACCEPTED_WITHOUT_REWARDED_REQUEST"],
        )


if __name__ == "__main__":
    unittest.main()
