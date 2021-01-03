import { Component, OnInit, ViewChild } from '@angular/core';
import { Track } from 'ngx-audio-player';
import { Subscription } from 'rxjs';
import { ApiService } from 'src/app/services/api.service';

@Component({
  selector: 'app-main-view',
  templateUrl: './main-view.component.html',
  styleUrls: ['./main-view.component.css'],
})
export class MainViewComponent implements OnInit {
  private sub = new Subscription();

  displayTitle = true;
  displayPlayList = true;
  pageSizeOptions = [2, 4, 6];
  displayVolumeControls = true;
  disablePositionSlider = true;

  playlist: Track[] = [];

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.updateSongsList();
  }

  updateSongsList(): void {
    this.sub = this.api.get().subscribe((res) => {
      var stringres = JSON.stringify(res);
      var jsonres = JSON.parse(stringres);
      this.playlist.length = 0;
      var songs = jsonres['songs']
      songs.forEach((element) => {
        this.playlist.push({ title: element['title'], link: element['url'] });
      });
    });
  }

  onEnded($event): void {}
}
