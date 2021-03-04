import os
import jieba
import numpy as np

from utils import read_json

class Turn:
    def __init__(self, role, utterance, dialog_act):
        self.role = role
        self.utterance = utterance
        self.dialog_act = dialog_act

        self.transcript = [word for word in jieba.cut(utterance)] if role == 'usr' else ''
        self.turn_label = [[action[0], action[1]] for action in dialog_act]
        self.belief_state = [{'slots':[action[1], action[2]], 'act':action[0]} for action in dialog_act]
        self.system_transcript = utterance if role == 'sys' else ''
        self.num = {}

    def to_dict(self):
        return {'transcript': self.transcript,
                'turn_label': self.turn_label,
                'belief_state': self.belief_state,
                'system_transcript': self.system_transcript,
                'num': self.num}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class Dialogue:
    def __init__(self, dialogue_id, turns):
        self.id = dialogue_id
        self.turns = turns

    def __len__(self):
        return len(self.turns)

    def to_dict(self):
        return {'dialogue_id': self.id,
                'turns': [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d):
        return cls(d['sys-usr'], [Turn.from_dict(t) for t in d['turns']])

class Dataset:
    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __len__(self):
        return len(self.dialogues)

    def iter_turns(self):
        for d in self.dialogues:
            for t in d.turns:
                yield t

    def to_dict(self):
        return {'dialogues': [d.to_dict() for d in self.dialogues]}

    @classmethod
    def from_dict(cls, d):
        return cls([Dialogue.from_dict(dd) for dd in d[:4]])

    def evaluate_preds(self, preds):
        request = []
        inform = []
        joint_goal = []
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        i = 0
        for d in self.dialogues:
            pred_state = {}
            for t in d.turns:
                gold_request = set([(s, v) for s, v in t.turn_label if s == 'Request'])
                gold_inform = set([(s, v) for s, v in t.turn_label if s != 'Request'])
                pred_request = set([(s, v) for s, v in preds[i] if s == 'Request'])
                pred_inform = set([(s, v) for s, v in preds[i] if s != 'Request'])
                request.append(gold_request == pred_request)
                inform.append(gold_inform == pred_inform)

                gold_recovered = set()
                pred_recovered = set()
                for s, v in pred_inform:
                    pred_state[s] = v
                for b in t.belief_state:
                    for s, v in b['slots']:
                        if b['act'] != 'Request':
                            gold_recovered.add((b['act'], fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())))
                for s, v in pred_state.items():
                    pred_recovered.add(('Inform', s, v))
                joint_goal.append(gold_recovered == pred_recovered)
                i += 1
        return {'turn_inform': np.mean(inform), 'turn_request': np.mean(request), 'joint_goal': np.mean(joint_goal)}

class Ontology:
    def __init__(self, path):
        self.slots = []
        self.values ={}
        self.num = {}

        files = os.listdir(path)

        for file in files:
            values = []
            for slot in read_json(path + "/" + file):
                value = slot[0]
                key = slot[1]['领域']
                values.append(value)
            self.values[key] = values
            self.slots.append(key)

    def to_dict(self):
        return {'slots': self.slots, 'values': self.values, 'num': self.num}
