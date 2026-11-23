import json
import unittest
from superagent.planning.llm import parse_plan_response


class ResponseTests(unittest.TestCase):
    def test_valid_reply_and_truncation_with_valid_looking_json(self):
        plan={'choice':0,'diagnosis':'Baseline measured','next_action':'Compare validation'}
        payload={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(plan)}}]}
        self.assertEqual(parse_plan_response(payload),plan)
        payload['choices'][0]['finish_reason']='length'
        with self.assertRaisesRegex(ValueError,'truncated'):parse_plan_response(payload)

    def test_malformed_provider_shapes_and_content(self):
        for payload in ({}, {'choices':[]}, {'choices':[None]}, {'choices':[{}]},
                        *({'choices':[{'message':{'content':content}}]} for content in (None,[], '', ' ', '{', '[]', 'null'))):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):parse_plan_response(payload)
