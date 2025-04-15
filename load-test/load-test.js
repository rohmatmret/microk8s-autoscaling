import http from 'k6/http';
import { sleep } from 'k6';

export let options = {
  stages: [
    { duration: '1m', target: 100 },    // Ramp-up
    { duration: '5m', target: 2000 },   // Traffic increased to 200%
    { duration: '1m', target: 0 },      // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200'],   // Target latency <200ms
  },
};

export default function () {
  http.get('http://<MICROK8S_IP>:<NODE_PORT>');
  sleep(1);
}