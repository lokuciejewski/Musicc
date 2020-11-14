import { Component, OnInit, ViewChild } from '@angular/core';
import { Track } from 'ngx-audio-player'

@Component({
  selector: 'app-main-view',
  templateUrl: './main-view.component.html',
  styleUrls: ['./main-view.component.css'],
})
export class MainViewComponent implements OnInit {
  displayTitle = true;
  displayPlayList = true;
  pageSizeOptions = [2, 4, 6];
  displayVolumeControls = true;
  disablePositionSlider = true;

  playlist: Track[] = [
    {
      title: 'Audio One Title',
      link: 'Link to Audio One URL'
    },
    {
      title: 'Audio Two Title',
      link: 'Link to Audio Two URL'
    },
    {
      title: 'Audio Three Title',
      link: 'Link to Audio Three URL'
    },
  ];

  constructor() {}

  ngOnInit(): void {}

  onEnded($event): void {}

}
