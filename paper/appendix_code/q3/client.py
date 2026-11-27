import http.client
import json
import math
import time
import urllib.error
import urllib.request
import uuid
from action_timing import movement_microseconds
from geometry import distance
class ProtocolError(RuntimeError):
    pass
class TimeBudgetError(RuntimeError):
    pass
class HttpTransport:
    def __init__(self, url):
        self.url = url.rstrip('/')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def send(self, path, payload, timeout):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        request = urllib.request.Request(self.url + path, data=body, headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return (response.status, json.loads(response.read().decode('utf-8')))
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                data = json.loads(raw.decode('utf-8'))
            except (ValueError, UnicodeError):
                data = {'accepted': False, 'error': 'Non-JSON HTTP error response'}
            return (error.code, data)
class Client:
    def __init__(self, transport, robot_id, log_path=None, wait_ready=0, timeout=5, retries=3):
        self.transport, self.robot_id = (transport, robot_id)
        self.wait_ready, self.timeout, self.retries = (wait_ready, timeout, retries)
        self.sequence, self.prefix = (0, uuid.uuid4().hex[:12])
        self.position, self.channel, self.virtual_time = ((0.0, 0.0), 1, 0.0)
        self.deadline = self.start = None
        self.max_virtual = 360000.0
        self.entered = self.exited = False
        self.counters = dict(measures=0, switches=0, clear_attempts=0, cleared=0, distance=0.0, movement_time_us=0, retries=0)
        self.journal = open(log_path, 'x', encoding='utf-8') if log_path else None
    def log(self, kind, **fields):
        if self.journal:
            self.journal.write(json.dumps(dict(kind=kind, time=time.time(), **fields), ensure_ascii=False, allow_nan=False) + '\n')
            self.journal.flush()
    def close(self):
        if self.journal:
            self.journal.close()
    def request(self, path, position=None, channel=None):
        if self.deadline is not None and time.monotonic() >= self.deadline - (0 if path == '/exit' else 2):
            raise TimeBudgetError('Reality deadline reached; completeness has NOT been established')
        if position is not None:
            if len(position) != 2 or any((not math.isfinite(x) or abs(x) > 2000000.0 for x in position)):
                raise ValueError('Invalid action coordinates')
            upper = movement_microseconds(distance(self.position, position)) / 1000000.0 + (6 if path == '/measure' else 5)
            if self.virtual_time + upper >= self.max_virtual:
                raise TimeBudgetError('Virtual budget would be exhausted')
        self.sequence += 1
        payload = dict(arena_id='default', robot_id=self.robot_id, request_id=f'{self.prefix}-{self.sequence}')
        if position is not None:
            payload.update(position=dict(x=float(position[0]), y=float(position[1])), channel=channel)
        self.log('request', path=path, payload=payload)
        sent_at = time.monotonic()
        ready_deadline = sent_at + self.wait_ready
        attempt = 0
        while True:
            timeout = self.timeout
            if self.deadline is not None:
                timeout = min(timeout, self.deadline - time.monotonic())
                if timeout <= 0:
                    raise TimeBudgetError('Deadline during request retry; action outcome may be unknown')
            try:
                status, result = self.transport.send(path, payload, timeout)
                if status in (429, 500, 502, 503, 504):
                    raise OSError(f'Temporary HTTP {status}')
            except (OSError, ValueError, http.client.HTTPException) as error:
                attempt += 1
                self.counters['retries'] += 1
                self.log('retry', request_id=payload['request_id'], error=str(error), attempt=attempt)
                if attempt > self.retries and (not (path == '/enter' and time.monotonic() < ready_deadline)):
                    raise ProtocolError(f'{path}: retry limit reached; action outcome may be unknown: {error}') from error
                time.sleep(0.2 if path != '/enter' else 0.5)
                continue
            self.log('response', request_id=payload['request_id'], status=status, response=result)
            if status != 200 or not isinstance(result, dict) or result.get('accepted') is not True:
                raise ProtocolError(f'{path} rejected: HTTP {status}, {result}')
            break
        vt = result.get('virtual_time_s')
        if type(vt) not in (int, float) or not math.isfinite(vt) or vt < self.virtual_time - 1e-06:
            raise ProtocolError('Invalid or decreasing virtual timestamp')
        self.virtual_time = float(vt)
        if path == '/enter':
            self.entered = True
            remaining = result.get('remaining_real_duration_s')
            if type(remaining) not in (int, float) or not math.isfinite(remaining) or (not 0 <= remaining <= 1200):
                raise ProtocolError('Missing/invalid remaining_real_duration_s')
            self.start = sent_at
            self.deadline = sent_at + remaining
            self.max_virtual = float(result.get('max_virtual_duration_s', 360000))
        elif path == '/exit':
            if result.get('exit_reason') != 'user_exit':
                raise ProtocolError('Unexpected exit response')
            self.exited = True
        else:
            length = distance(self.position, position)
            self.counters['distance'] += length
            self.counters['movement_time_us'] += movement_microseconds(length)
            self.position = tuple(position)
            if path == '/measure':
                self.counters['measures'] += 1
                self.counters['switches'] += int(channel != self.channel)
                self.channel = channel
                if result.get('measure_result') not in ('near', 'direction', 'no_signal'):
                    raise ProtocolError('Unknown measurement result')
                if result['measure_result'] == 'direction':
                    angle = result.get('svd_deg')
                    if type(angle) not in (int, float) or not math.isfinite(angle) or (not 0 <= angle < 360):
                        raise ProtocolError('Invalid bearing')
            else:
                self.counters['clear_attempts'] += 1
                if result.get('clear_result') not in ('success', 'no_target_in_range'):
                    raise ProtocolError('Unknown clear result')
                self.counters['cleared'] += int(result['clear_result'] == 'success')
        return result
