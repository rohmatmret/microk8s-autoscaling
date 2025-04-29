import http from 'k6/http';
import { sleep } from 'k6';

export let options = {
  stages: [
    { duration: '1m', target: 100 },    // Ramp-up
    // { duration: '5m', target: 2000 },   // Traffic increased to 200%
    { duration: '1m', target: 0 },      // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200'],   // Target latency <200ms
  },
};

export default function () {
  check(res, {
    'is status 200': (r) => r.status === 200,
});

console.log('Starting request at: ' + new Date().toISOString());

    let res = http.get('http://nginx-service:80');
    
    console.log(`Response received at: ${new Date().toISOString()}`);
    console.log(`Response time: ${res.timings.duration}ms`);
    
    check(res, {
        'is status 200': (r) => r.status === 200,
    });
  // http.get('http://nginx-service.default.svc.cluster.local'); // access Internal CluterIP
  // sleep(1);
}