import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { AppSettings } from 'src/app/app.settings';
@Injectable({
  providedIn: 'root',
})
export class ApiService {
  constructor(private httpClient: HttpClient) {}

  get(): Observable<Object> {
    return this.httpClient.get(AppSettings.API_ENDPOINT);
  }

  post(args: any): Observable<Object> {
    return this.httpClient.post(AppSettings.API_ENDPOINT, args);
  }
}
