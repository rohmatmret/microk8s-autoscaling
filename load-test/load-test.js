import http from 'k6/http';
import { sleep } from 'k6';

export let options = {
  stages: [
    { duration: '1m', target: 1000 },    // Ramp-up
    { duration: '1m', target: 2000 },   // Sustain 200 VUs
    { duration: '1m', target: 6000 },   // Traffic increased to 200%
    { duration: '1m', target: 10000 },   // Traffic increased to 200%
    { duration: '1m', target: 0 },      // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200'],   // Target latency <200ms
  },
};

export default function () {
  
  http.get('http://192.168.64.8:30597'); // access Internal CluterIP
  sleep(1);
}